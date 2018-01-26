clear all;
datasets = {'MLGene/'};
ML = 1; % 0 -multi-class and 1-multi-label
train_percentage = [0.02 0.04 0.06 0.08 0.1];
algorithm = 2; %1 -- svm 2 -- NB
K = 3;
max_iter = 20;

for d = 1:size(datasets,2)
    [view,links] = base2_prepare_sv_ml(datasets);        
    n_ids = size(view,1);  
    n_views = size(links,2);
   
    load(char(strcat(datasets(d),'raw_ids.mat')));%loads ids
    truth_labels = load(char(strcat(datasets(d),'truth.mat'))); %loads truth(-1,1) class
    truth_labels = truth_labels.('truth');   
    n_labels = size(truth_labels,2);
    truth_labels(truth_labels == -1) = 0;
    
    for train_perc = train_percentage
        fid = fopen(char(strcat(datasets(d),'Base2_results_',num2str(train_perc*100),'.txt')),'w');

        disp(train_perc);
        accuracy = zeros(K,1);
        h_accuracy = zeros(K,1);
        precision = zeros(K,1);
        recall = zeros(K,1);
        f_measure = zeros(K,1);
        
       %run ICA with cross validation by 10 folds
       
       for k = 1:K
         
         disp(strcat('K ============',num2str(k)));
         load(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100),'/',num2str(k),'.mat')));         
         unlabelled_indices  = ~labelled_indices;

         n_train = nnz(labelled_indices);
         n_test = nnz(unlabelled_indices);
         
         nf = size(view,2);
         tot_nf = nf + n_labels;
         view_link_features = zeros(n_train,tot_nf);
         
         views = struct;             
         views(1).svm = struct;
                 
         %TRAINING:        
         for label_id = 1:n_labels                  
            for view_id = 1:n_views
                 view_link_features = [view(labelled_indices,:) base2_buildrelation(links(view_id).adjmat(labelled_indices,labelled_indices),truth_labels(labelled_indices,:))];
                  %views(view_id).svm(label_id).model = svmtrain(truth_labels(labelled_indices,label_id),view_link_features,'-t 2 -b 1');
                 %views(view_id).svm(label_id).model = ClassificationTree.fit(view_link_features,truth_labels(labelled_indices,label_id));
                 views(view_id).svm(label_id).model =  NaiveBayes.fit(view_link_features,truth_labels(labelled_indices,label_id),'Distribution','mn');
            end
         end    
         %disp('Training Over');
         
         test = find((1:n_ids)'.*(unlabelled_indices));
         predicted_labels = truth_labels;
                  
          %BOOTSTRAP:
         for test_id = test'
             PosterPos = zeros(1,n_labels);
             PosterNeg = zeros(1,n_labels);                
             for label_id = 1:n_labels
                 for view_id = 1:n_views
                    %[~,~,prob_estimates] = svmpredict(ones(1,1), [view(test_id,:) zeros(1,n_labels)], views(view_id).svm(label_id).model,'-b 1');
                    %PosterPos = PosterPos + prob_estimates(1,2);                    
                    %PosterNeg = PosterNeg + prob_estimates(1,1);                    
                     
                    [score] = posterior( views(view_id).svm(label_id).model,[view(test_id,:) zeros(1,n_labels)]);
                    PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,find(views(view_id).svm(label_id).model.ClassLevels == 1));
                    PosterNeg(:,label_id) = PosterNeg(:,label_id) + ( 1 - score(:,find(views(view_id).svm(label_id).model.ClassLevels == 1)));
                 end                 
             end
            PosterPos = PosterPos / n_views;
            PosterNeg = PosterNeg / n_views;
            if ML == 0
                [~,tid] = max(PosterPos);
                predicted_labels(test_id,tid) = 1;
            else
                predicted_labels(test_id,:) = PosterPos >= PosterNeg;
            end
         end
         %disp('Bootstrap Over');
         
         
         %ITERATIVE INFERENCE:
         for iter = 1:max_iter
             %disp(iter);             
             o_predicted_labels = predicted_labels;
             for test_id = test'
                PosterPos = zeros(1,n_labels);
                PosterNeg = zeros(1,n_labels);
                for label_id = 1:n_labels                    
                    for view_id = 1:n_views
                        %Build updated Relational Data
                        view_link_features = [ view(test_id,:) base2_buildrelation(links(view_id).adjmat(test_id,:),o_predicted_labels)];         
                        
                        %Inference                        
                        %[~,~,prob_estimates] = svmpredict(ones(1,1), view_link_features, views(view_id).svm(label_id).model,'-b 1');
                        %PosterPos = PosterPos + prob_estimates(1,2);                    
                        %PosterNeg = PosterNeg + prob_estimates(1,1);                    
                        [score] = posterior( views(view_id).svm(label_id).model,[view(test_id,:) zeros(1,n_labels)]);
                        PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,find(views(view_id).svm(label_id).model.ClassLevels == 1));
                        PosterNeg(:,label_id) = PosterNeg(:,label_id) + ( 1 - score(:,find(views(view_id).svm(label_id).model.ClassLevels == 1)));   
                     end
                end
                PosterPos = PosterPos / n_views;
                PosterNeg = PosterNeg / n_views;
                if ML == 0
                    [~,tid] = max(PosterPos);
                    predicted_labels(test_id,tid) = 1;
                else
                    predicted_labels(test_id,:) = PosterPos >= PosterNeg;
                end
             end         
         end
         disp('Inference over');
         [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)] = calc_acc_CoTraining(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));             
         fprintf(fid,'%f %f %f %f %f\n',[accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)]);         
       end
       disp([accuracy precision recall f_measure h_accuracy]);
       fprintf(fid,'%f %f %f %f %f\n',[mean(accuracy) mean(precision) mean(recall) mean(f_measure) mean(h_accuracy)]);
       fclose(fid);
    end
end
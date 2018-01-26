clear all;
datasets = {'MLGene/'};
ML = 1; % 0 -multi-class and 1-multi-label
train_percentage = [0.02 0.04 0.06 0.08 0.1];
%train_percentage = 0.02;
algorithm = 1; %1 -- classifier 2 -- NB
K = 3;
max_iter = 1;

for d = 1:size(datasets,2)
    [view,links] = base1_prepare_sv_ml(datasets);        
    n_ids = size(view,1);  
    n_views = size(links,2);
   
    load(char(strcat(datasets(d),'raw_ids.mat')));%loads ids
    truth_labels = load(char(strcat(datasets(d),'truth.mat'))); %loads truth(-1,1) class
    truth_labels = truth_labels.('truth');   
    n_labels = size(truth_labels,2);
    truth_labels(truth_labels == -1) = 0;
    
    for train_perc = train_percentage
        fid = fopen(char(strcat(datasets(d),'Base1_results_',num2str(train_perc*100),'.txt')),'w');

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
         views(1).classifier = struct;
                 
         %TRAINING:        
         for label_id = 1:n_labels                  
            for view_id = 1:n_views
                 view_link_features = [view(labelled_indices,:) base1_buildrelation(links(view_id).adjmat(labelled_indices,labelled_indices),truth_labels(labelled_indices,:))];
                  %views(view_id).classifier(label_id).model = classifiertrain(truth_labels(labelled_indices,label_id),view_link_features,'-t 2 -b 1');
                 views(view_id).classifier(label_id).model = ClassificationTree.fit(view_link_features,truth_labels(labelled_indices,label_id));
                 views(view_id).classifier(label_id).modela = ClassificationTree.fit(view(labelled_indices,:),truth_labels(labelled_indices,label_id));
                 %views(view_id).classifier(label_id).model =  NaiveBayes.fit(view_link_features,truth_labels(labelled_indices,label_id),'Distribution','mn');
                 %views(view_id).classifier(label_id).modela =  NaiveBayes.fit(view(labelled_indices,:),truth_labels(labelled_indices,label_id),'Distribution','mn');
            end
         end    
         %disp('Training Over');
         
         
         test = find((1:n_ids)'.*(unlabelled_indices));
         predicted_labels = truth_labels;
         predicted_labels(unlabelled_indices,:) = zeros(n_test,n_labels);
                  
         train = find((1:n_ids)'.*(labelled_indices));
         ocnt = zeros(length(test),1);
         cnt = 1;
         for test_id = test'
            ocnt(cnt,1) = length(find(links(1).adjmat(test_id,:)));
            cnt = cnt + 1;
         end         
         tmp = sum(links(1).adjmat')';
         ocnt = tmp(unlabelled_indices,1);
         %BOOTSTRAP:
         [a,b] = sort(ocnt,'descend');
         test = test(b);
         pos_index = find(views(view_id).classifier(label_id).modela.ClassNames == 1,1);
         
          %BOOTSTRAP:
         for test_id = test'
             PosterPos = zeros(1,n_labels);
             PosterNeg = zeros(1,n_labels);                
             for label_id = 1:n_labels
                 for view_id = 1:n_views
                    %[~,~,prob_estimates] = classifierpredict(ones(1,1), [view(test_id,:) zeros(1,n_labels)], views(view_id).classifier(label_id).model,'-b 1');
                    %PosterPos = PosterPos + prob_estimates(1,2);                    
                    %PosterNeg = PosterNeg + prob_estimates(1,1);                    
                     
                    [~,score] = predict(views(view_id).classifier(label_id).modela,view(test_id,:));                    
                    PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,pos_index);
                    PosterNeg(:,label_id) = PosterNeg(:,label_id) + (1 - score(:,pos_index));

                    
                    %[score] = posterior( views(view_id).classifier(label_id).modela,view(test_id,:));
                    %PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,find(views(view_id).classifier(label_id).model.ClassLevels == 1));
                    %PosterNeg(:,label_id) = PosterNeg(:,label_id) + ( 1 - score(:,find(views(view_id).classifier(label_id).model.ClassLevels == 1)));
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
             %o_predicted_labels = predicted_labels;
             for test_id = test'
                PosterPos = zeros(1,n_labels);
                PosterNeg = zeros(1,n_labels);
                for label_id = 1:n_labels                    
                    for view_id = 1:n_views
                        %Build updated Relational Data
                        view_link_features = [ view(test_id,:) base1_buildrelation(links(view_id).adjmat(test_id,:),predicted_labels)];         
                        
                        %Inference                        
                        %[~,~,prob_estimates] = classifierpredict(ones(1,1), view_link_features, views(view_id).classifier(label_id).model,'-b 1');
                        %PosterPos = PosterPos + prob_estimates(1,2);                    
                        %PosterNeg = PosterNeg + prob_estimates(1,1);                    
     
                        [~,score] = predict(views(view_id).classifier(label_id).modela,view(test_id,:));                    
                        PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,pos_index);
                        PosterNeg(:,label_id) = PosterNeg(:,label_id) + (1 - score(:,pos_index));
                        
                        %[score] = posterior( views(view_id).classifier(label_id).model,[view(test_id,:) zeros(1,n_labels)]);
                        %PosterPos(:,label_id) = PosterPos(:,label_id) + score(:,find(views(view_id).classifier(label_id).model.ClassLevels == 1));
                        %PosterNeg(:,label_id) = PosterNeg(:,label_id) + ( 1 - score(:,find(views(view_id).classifier(label_id).model.ClassLevels == 1)));   
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
              if ML == 0
                  [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1)] = calc_acc_CoTrainingmc(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));
              else
                  [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)] = calc_acc_CoTrainingml(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));
              end 
              disp([k iter f_measure(k,1)])
         end
         disp('Inference over');
         %[accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)] = calc_acc_CoTraining(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));             
          if ML == 0
              [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1)] = calc_acc_CoTrainingmc(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));
          else
              [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)] = calc_acc_CoTrainingml(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));
          end 
         fprintf(fid,'%f %f %f %f %f\n',[accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1)]);         
       end
       disp([accuracy precision recall f_measure h_accuracy]);
       fprintf(fid,'%f %f %f %f %f\n',[mean(accuracy) mean(precision) mean(recall) mean(f_measure) mean(h_accuracy)]);
       fclose(fid);
    end
end
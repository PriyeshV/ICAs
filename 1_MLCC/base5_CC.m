clear all;
datasets = {'MCGene/'};
ML = 0; % 0 -multi-class and 1-multi-label
train_percentage = [0.05 0.1 0.15 0.2 0.25];
algorithm = 1; %1 -- svm 2 -- NB
K = 3;
max_iter = 20;

for d = 1:size(datasets,2)
    
   %combine all view to single view and all links to single link
   [view,link] = base5_prepare_sv_sl(datasets(d));
   n_ids = size(view,1);  
   
   load(char(strcat(datasets(d),'raw_ids.mat')));%loads ids
   truth_labels = load(char(strcat(datasets(d),'truth.mat'))); %loads truth(-1,1) class
   truth_labels = truth_labels.('truth');   
   n_labels = size(truth_labels,2);
   truth_labels(truth_labels == -1) = 0;
   
   for train_perc = train_percentage
        fid = fopen(char(strcat(datasets(d),'Base5_results_',num2str(train_perc*100),'.txt')),'w');
          disp(train_perc);
        accuracy = zeros(K,1);
		ex_accuracy = zeros(K,1);
        h_accuracy = zeros(K,1);
        precision = zeros(K,1);
        recall = zeros(K,1);
        f_measure = zeros(K,1);
		label_acc = zeros(K,1);
      
       %run ICA with cross validation by 10 folds
       
       for k = 1:K
         
         disp(strcat('K ============',num2str(k)));
         load(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100),'/',num2str(k),'.mat')));
         unlabelled_indices  = ~labelled_indices;

         n_train = nnz(labelled_indices);
         n_test = nnz(unlabelled_indices);
         a_nf = size(view,2);
         tot_nf = a_nf + (3*n_labels-1);

         correct_rate = 0;
         cperf = struct; 
         svm = struct;

         %TRAINING:        
         for label_id = 1:n_labels
             view_link_features = zeros(n_train,tot_nf);         

             view_link_features(:,1:a_nf) = view(labelled_indices,:);
             view_link_features(:,(a_nf+1):(a_nf+(n_labels-1))) = truth_labels(labelled_indices,setdiff(1:n_labels,label_id));

             link_features  = base5_buildrelation(link(labelled_indices,labelled_indices),truth_labels(labelled_indices,:),label_id);
             view_link_features(:,a_nf+n_labels:tot_nf) = link_features;

             %svm(label_id).model = libsvmtrain(truth_labels(labelled_indices,label_id),view_link_features,'-t 2 -b 1');
            
             svm(label_id).model =  NaiveBayes.fit(view_link_features,truth_labels(labelled_indices,label_id),'Distribution','mn');
         end    
         %disp('Training Over');

         test = find((1:n_ids)'.*(unlabelled_indices));
         predicted_labels = zeros(size(truth_labels));
         
         %BOOTSTRAP:
         for test_id = test'
             PosterPos = zeros(1,n_labels);
             PosterNeg = zeros(1,n_labels); 
             for label_id = 1:n_labels            
                %[predicted_labels(test_id,label_id),acc,prob_estimates] = libsvmpredict(ones(1,1), [view(test_id,:) zeros(1,3*n_labels-1)], svm(label_id).model,'-b 1');
                [score] = posterior(svm(label_id).model,[view(test_id,:) zeros(1,3*n_labels-1)]);
                PosterPos(:,label_id) = score(:,find(svm(label_id).model.ClassLevels == 1));
                PosterNeg(:,label_id) = ( 1 - score(:,find(svm(label_id).model.ClassLevels == 1)));
             end
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
             predicted_labels = zeros(size(o_predicted_labels));
             for test_id = test'
                 for label_id = 1:n_labels

                     %Build updated Relational Data
                     view_link_features = zeros(1,tot_nf);
                     view_link_features(1,1:a_nf) = view(test_id,:);
                     view_link_features(1,(a_nf+1):(a_nf+(n_labels-1))) = o_predicted_labels(test_id,setdiff(1:n_labels,label_id));
                     link_features  = base5_buildrelation(link(test_id,:),o_predicted_labels,label_id);
                     view_link_features(:,a_nf+n_labels:tot_nf) = link_features;

                     %Inference
                    % predicted_labels(test_id,label_id) = svmclassify(svm(label_id).model,view_link_features);
                    %[predicted_labels(test_id,label_id),acc,prob_estimates] = libsvmpredict(ones(1,1), view_link_features, svm(label_id).model,'-b 1');
                    [score] = posterior(svm(label_id).model,[view(test_id,:) zeros(1,3*n_labels-1)]);
                    PosterPos(:,label_id) = score(:,find(svm(label_id).model.ClassLevels == 1));
                    PosterNeg(:,label_id) = ( 1 - score(:,find(svm(label_id).model.ClassLevels == 1)));
                 end
                 if ML == 0
                     [~,tid] = max(PosterPos);
                     predicted_labels(test_id,tid) = 1;
                 else
                     predicted_labels(test_id,:) = PosterPos >= PosterNeg;
                 end
             end
             
            old = [accuracy(k,1) precision(k,1) recall(k,1) f_measure(k,1) h_accuracy(k,1) ex_accuracy(k,1) label_acc(k,1)];
            [accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1),ex_accuracy(k,1),label_acc(k,1)] = calc_acc_CoTraining(truth_labels(unlabelled_indices,:),predicted_labels(unlabelled_indices,:));             
            new = [accuracy(k,1) precision(k,1) recall(k,1) f_measure(k,1) h_accuracy(k,1) ex_accuracy(k,1) label_acc(k,1)];
            if sum(old == new) == 7
                break;
            end             
         end
         disp('Inference over');         
         fprintf(fid,'%f %f %f %f %f %f %f\n',[accuracy(k,1),precision(k,1),recall(k,1),f_measure(k,1),h_accuracy(k,1),ex_accuracy(k,1),label_acc(k,1)]);
       end
       disp([accuracy precision recall f_measure h_accuracy ex_accuracy label_acc]);
       fprintf(fid,'%f %f %f %f %f %f %f\n',[mean(accuracy) mean(precision) mean(recall) mean(f_measure) mean(h_accuracy) mean(ex_accuracy) mean(label_acc)]);
       fclose(fid);
   end
end
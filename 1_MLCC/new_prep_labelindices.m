datasets = {'Data/'};
train_percentage = [0.1 0.3 0.5 0.7 0.9];
K = 5;%5 fold validation
k = 1;
for train_perc = train_percentage
    for d = 1:size(datasets,2)
        datasets(d)
        mkdir(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100))));
        %load ids
        load(char(strcat(datasets(d),'raw_ids.mat')));
        n_ids = size(ids,1);

        %load truth labels
        truth = load(char(strcat(datasets(d),'truth.mat')));
        truth = truth.('truth');
        n_labels = size(truth,2);
        truth(truth == -1) = 0;
        
        n_pos = sum(truth);
        n_neg = n_ids - n_pos;
        pos_ratio = n_pos ./ n_ids;
        n_labelled = ceil(train_perc * n_ids);
        n_pos_req = floor(n_labelled .* pos_ratio);
        n_pos_req(n_pos_req == 0) = 1;
        [reqd,labels] = sort(n_pos_req,'descend');
        
        for k = 1:K
            labelled_indices = false(n_ids,1);
            for label_id = labels
                p_posi = find(truth(:,label_id));
                n_p_posi = size(p_posi,1);
                ind = zeros(1,0);
                curr = find(labelled_indices');
                while size(ind,2) < (n_pos_req(1,label_id) - nnz(truth(labelled_indices,label_id)))
                    new = p_posi(randi([1 n_p_posi]));
                    if isempty(find(new == find(labelled_indices), 1))
                        ind = unique([ind new ]);
                    end
                end
                labelled_indices(ind,1) = 1;
            end
            save(char(strcat(datasets(d),'labelled_indices_perc_',num2str(train_perc*100),'/',num2str(k),'.mat')),'labelled_indices'); 
        end
    end
end

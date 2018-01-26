function [relationalmat] = base2_buildrelation(neighborhood,labels)
    
    n_ids = size(neighborhood,1);
        
    n_labels = size(labels,2);
    relationalmat = zeros(n_ids,n_labels);

    for i=1:n_ids               
        n_neighborhood = nnz(neighborhood(i,:));
        rel_features = sum([ labels(logical(neighborhood(i,:)),:); zeros(2,n_labels)]);
        rel_features = rel_features ./ n_neighborhood;
        
        relationalmat(i,:) = rel_features;
    end
end 
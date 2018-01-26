function [relationalmat] = base1_buildrelation(neighborhood,labels,label_id)
    
    n_ids = size(neighborhood,1);
        
    n_labels = size(labels,2);
    relationalmat = zeros(n_ids,2*n_labels);

    for i=1:n_ids               
        SL_neighborhood = logical(neighborhood(i,:) .* labels(:,label_id)');     
        n_SL_neighborhood = nnz(SL_neighborhood);
        SL_rel_features = sum([ labels(SL_neighborhood,:); zeros(2,n_labels)]);
        %SL_rel_features = SL_rel_features ./ n_SL_neighborhood;
        
        CL_neighborhood = logical(neighborhood(i,:) .* (~labels(:,label_id))');        
        n_CL_neighborhood = nnz(CL_neighborhood);
        CL_rel_features = sum([ labels(CL_neighborhood,:); zeros(2,n_labels)]);
        %CL_rel_features = CL_rel_features ./ n_CL_neighborhood;
        
        relationalmat(i,:) = [ SL_rel_features CL_rel_features];
    end
end 
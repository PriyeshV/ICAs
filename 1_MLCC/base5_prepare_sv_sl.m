function [view,link ] = base0_prepare_sv_sl(dataset)

    %name_views = {'Tags','Summary'};
    %name_views = {'Summary','Description'};            
    name_views = {'view'};
    
    %name_links = {'Reporter_link'};
    %name_links = {'Actor_Graph','Director_Graph'};
    name_links = {'adjmat1','adjmat2'};

    load(char(strcat(dataset,'/raw_ids.mat')));
    n_ids = size(ids,1);
    view = zeros(n_ids,0);
    link = zeros(n_ids,n_ids);
    
    for v = 1:size(name_views,2)
        file_name = char(strcat(dataset,name_views(v),'.mat'));
        tmp = load(file_name);
        tmp = tmp.('view');
        view = [view tmp];
    end
        
    for l = 1:size(name_links,2)
        file_name = char(strcat(dataset,name_links(l),'.mat'));
        tmp = load(file_name);
        tmp = tmp.('links');
        link = link + tmp ;      
    end
    link(link>0) = 1;
    for i = 1:n_ids
        for j = i+1:n_ids
            if link(i,j) == 1
                link(j,i) = 1;
            elseif link(j,i) == 1
                link(i,j) = 1;
            end
        end
    end
end


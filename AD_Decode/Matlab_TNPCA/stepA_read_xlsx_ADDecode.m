clear

%Original location of this file is
% /Users/jas/MATLAB/popNet_HOPCA_ADDecode_2024/ourscripts
%was also moved to DTC_private/AD_Decode/Matlab

filename = "/Users/jas/AD_Decode/CSVfiles/AD_DECODE_data3.xlsx" ;

test = 0;
act = 1;

mainpath =  "/Users/jas/MATLAB/popNet_HOPCA_ADDecode_2024/";

variant = ["fa"]; %Variant can be 'plain': plain connectome, 'dist': distance connectome, or 'fa': fa connectome
%please separate folders for specific connectomes before use

stripconnectivity=1; %True / False variable to determine whether connections with very low values should
%be removed entirely (see end of file)

if test
    connectomes_folder = "/Users/jas/jacques/Figures_temp_test/Figures_MPCA_inclusive_symmetric_all/";
    datapath = join([mainpath, "/data_test/"],"");
else
    if act
        plainpath = join([mainpath, "/conn_plain_act/"],"");
        connectomes_folder = "/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/connectome_act/";
        if variant == 'dist'
            datapath = join([mainpath, "/distances_act/"],"");
        elseif variant == 'fa'
            datapath = join([mainpath, "/famean_act/"],"");
        elseif variant == 'plain'
            datapath = plainpath;
        end
    else
        connectomes_folder = "/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/connectome/";
        if variant == 'dist'
            datapath = join([mainpath, "/distances/"],"");
        elseif variant == 'fa'
            datapath = join([mainpath, "/famean/"],"");
        elseif variant == 'plain'
            datapath = join([mainpath, "/conn_plain/"],"");
        end

    end
end


size_var = size(variant);
if size_var(1)>1
    variant_str = '_' + strjoin(variant,'_');
else
    %variant_str = '';
    if strcmp(variant,'plain')
        variant_str = '';
    else
        variant_str = join(['_' variant],"");
    end
end


if ~exist(datapath, 'dir')
   mkdir(datapath)
end

%subselect is used to only select those connectomes associiated with a subject with certain characteriscs.
%If empty, include everyone. _genotype_4 => only APOE4 (only functional for
%genotype at this time
%subselect has not been used for a while and may have conflicts with the
%connectome strip function (see end of file)
subselect = ''; %subselect = '_genotype_4';  

if test
    getpath = join([connectomes_folder,'*',variant_str,'_connectomes.xlsx'],"");
else
    if variant=='fa'
        getpath = join([connectomes_folder,'*','mean_FA_connectome.csv'],"");
    elseif variant == 'dist'
        getpath = join([connectomes_folder,'*',variant_str,'distances.csv'],"");
    elseif variant == 'plain'
        getpath = join([connectomes_folder,'*',variant_str,'conn_plain.csv'],"");
    end
end

%Output paths, named differently if doing a subselection or on connectome
%type
response_array_path = join([datapath 'response_array' subselect variant_str '.mat'],"");
response_table_path= join([datapath 'response_table' subselect variant_str '.mat'],""); 
connectomes_path = join([datapath 'connectivity' '_ADDecode' '_mrtrix' subselect variant_str '.mat'],"");

%Read the metadata for this project
allfiles_data = readtable(filename);
allfiles_data = allfiles_data(~(isnan(allfiles_data.MRI_Exam) | strcmp(allfiles_data.MRI_Exam, '') | allfiles_data.MRI_Exam == 0), :);


%Create a table storing simplified metadata in Matlab Table, array
genotypekeySet = {'APOE33', 'APOE23', 'APOE34', 'APOE44'};
genotypevalueSet = [3 3 4 4];
geno = containers.Map(genotypekeySet,genotypevalueSet);

sexkeySet = {'M', 'F'};
sexvalueSet = [1 2];
sex = containers.Map(sexkeySet,sexvalueSet);

response_table = table('Size',[size(allfiles_data,1) 5],'VariableNames',{'Subject', 'sex', 'MRI_Exam', 'age', 'genotype'}, 'VariableTypes',{'string','string','double', 'double','string'});
response_array_init = zeros([size(allfiles_data,1) 4]);
for i = 1:size(allfiles_data,1)
    try
        response_array_init(i,2) = geno(cell2mat(table2array(allfiles_data(i,'genotype'))));
    catch exception
        continue
    end
    response_array_init(i,1) = table2array(allfiles_data(i,'MRI_Exam'));
    response_array_init(i,3) = table2array(allfiles_data(i,'age'));
    response_array_init(i,4) = sex(cell2mat(table2array(allfiles_data(i,'sex'))));
    response_table(i,1) = num2cell(table2array(allfiles_data(i,'Subject')));
    response_table(i,2) = num2cell(table2array(allfiles_data(i,'sex')));
    response_table(i,3) = num2cell(table2array(allfiles_data(i,'MRI_Exam')));
    response_table(i,4) = num2cell(table2array(allfiles_data(i,'age')));
    response_table(i,5) = num2cell(geno(cell2mat(table2array(allfiles_data(i,'genotype')))));
end


%Extracting all connectomes in a loop (IIT mean connectome of 84x84 is
%assumed)
connectivity = zeros(84,84,size(response_table,1));
files = dir(getpath);
subjlist = zeros(size(response_table,1),1); %List of subjects with associated connectomes
notfoundlist = zeros(size(response_table,1),1); %List of subjects not found
j=1;
l=1;
for i = 1:size(response_table,1)
    subjname = response_table{i,3};
    found = 0;
    for file = files'
        subj = strsplit(file.name,'_');
        subj = subj{1};
        subj = subj(3:end);
        if subjname == str2double(subj)
            csv = readtable(join([connectomes_folder,file.name],""));
            if test
                csv.Var1{84} = 'ctx-rh-insula';
                csv = removevars(csv,{'Var1'});
            end
            csv = table2array(csv);
            connectivity(:,:,j) = csv;
            found = 1;
            break
        end
    end
    if found==1
        %display('found '+ string(subjname))
        subjlist(j) = subjname;
        j = j + 1;
    else
        %display('did not find '+ string(subjname))
        notfoundlist(l) = subjname;
        l = l + 1;
    end
end

%Creating final array, only including connectomes that were found (in the
%same order as the connectome array itself)
subjlist = subjlist(subjlist ~= 0);
notfoundlist = notfoundlist(notfoundlist ~= 0);
connectivity = connectivity(:,:,1:size(subjlist,1));
response_array = zeros(size(subjlist,1),size(response_array_init,2));
i=1;

for k = 1:numel(response_array_init)
    if ismember(response_array_init(k),subjlist)
        index_subj = find(subjlist == response_array_init(k));
        response_array(index_subj,:) = response_array_init(k,:);
    end
end


%Doign the subselection based on subselect
if size(subselect,2)>0
    if contains(subselect,'genotype')
        if contains(subselect,'3')
            APOE3 = response_array(:,2)==3;
            connectivity = connectivity(:,:,APOE3);
            response_array = response_array(APOE3,:);
            subjlist = subjlist(APOE3);
        elseif contains(subselect,'4')
            APOE4 = response_array(:,2)==4;
            connectivity = connectivity(:,:,APOE4);
            response_array = response_array(APOE4,:);
            subjlist = subjlist(APOE4);
        end
    end
end

nan_indices = [];

n_elements = size(connectivity, 3);
nan_indices = [];

% Iterate through the third dimension
for n = 1:n_elements
    % Check if any element in the 2D matrix at index n is NaN
    if any(any(isnan(connectivity(:,:,n))))
        % If NaN is found, store the index
        nan_indices = [nan_indices, n];
    end
end
subjects_failed = response_array(nan_indices,1);
if ~isempty(subjects_failed)
    disp('List of failed subjects with Nan values, removed')
    disp(subjects_failed)
end

response_array = response_array(setdiff(1:n_elements, nan_indices),:);
connectivity = connectivity(:,:,setdiff(1:n_elements, nan_indices));

%Plain connectome is saved a bit differently and separately due to it being
%used by other variants for connectome strip setup, recommended to run plain for everyone first
connectomes_path_backup = join([plainpath 'connectivity' '_ADDecode' '_mrtrix' subselect '_plain_backup.mat'],"");
if variant == 'plain' && ~isfile(connectomes_path_backup)
    connectomes_path_backup = join([datapath 'connectivity' '_ADDecode' '_mrtrix' subselect variant_str '_backup.mat'],"");
    save(connectomes_path_backup, 'connectivity', 'subjlist');
end

%Connectome strip, removing indices where the average is lower than ten
%connections, or where too many subjects have an index value that is less
%than to
if stripconnectivity && variant~='plain'
    
    connectomes_plain_backup = load(connectomes_path_backup);
    connectivity_plain = connectomes_plain_backup.('connectivity');
    subjlist_plain = connectomes_plain_backup.('subjlist');
    if ~all(subjlist==subjlist_plain)
        throw('subjlist is different, redo plain connectome')
    end
    
    % Assuming your 3D matrix is named 'data'
    % N is the size of the third dimension

    % Step 1: Calculate the average for each ixj slice over the third dimension
    average_values = mean(connectivity_plain, 3);

    % Step 2: Identify indices where the average value is lower than 10
    low_average_indices = average_values < 10;

    % Step 3: Count instances where ixj has value 0 in the third dimension
    zero_count = sum(connectivity_plain == 0, 3);

    % Step 4: Identify indices where there are already 10 or more instances of ixj being 0
    too_many_zeros_indices = zero_count >= 2;

    % Step 5: Identify indices where either condition is true
    indices_to_remove = low_average_indices | too_many_zeros_indices;

    % Step 6: Set values to 0 for the identified indices
    num_subjs = size(connectivity,3);
    connectivity_new = zeros(size(connectivity));
    connectivity_plain_new = zeros(size(connectivity));
    for subj = 1:num_subjs
        connectivity_new(:, :, subj) = connectivity(:, :, subj) .* ~indices_to_remove;
        connectivity_plain_new(:, :, subj) = connectivity_plain(:, :, subj) .* ~indices_to_remove;
    end
    connectivity = connectivity_new;
end

save(response_array_path, 'response_array');
%save(response_table_path, 'response_table');
save(connectomes_path, 'connectivity', 'subjlist');


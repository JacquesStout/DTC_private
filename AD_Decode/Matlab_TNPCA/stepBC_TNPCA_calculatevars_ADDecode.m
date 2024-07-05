%-------------------------------Script Information-------------------------
% This script is used for extrcting Principle Components by Tensor PCA and
% also help you decide how many PCs to use
% Author: Jacques Stout & Wenlin Wu
% Last change:2024-07-4
% Change: cleanup and commenting update

%-------------------------------Output Information-------------------------
%U: subject mode, store PC score for the data;
%V: network modem store network basis;
%percent_store: store variation information that top PCs can reflect

%-------------------------------Setup Parameter----------------------------
clear

addpath('/Users/jas/MATLAB/tensor_toolbox')

mainpath =  "/Users/jas/MATLAB/popNet_HOPCA_ADDecode_2024/";

extended_analysis = 0;

act = 1;

%if act
%    datapath = join([mainpath, "/distances_act/"],"");
%    outpath =join([mainpath '/results_distances_act/'],"");
%else
%    datapath = join([mainpath, "/distances/"],"");
%end

%variants = ['*'];

%variant = ['plain'];
variant = 'plain' ;
analysis_type = 'age'; %genotype, sex, age

%variants = ["volweighted","fa"];
size_var = size(variant);
if size_var(1)>1
    variants_str = '_' + strjoin(variant,'_');
else
    %variants_str = '';
    if strcmp(variant,'plain')
        variant_str = '';
    else
        variant_str = join(['_' variant],"");
    end
end


%Should be kept consistent with stepA code, 
%clarifies the connectome type, 'plain', 'fa', 'distances', etc

if act
    connectomes_folder = "/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/connectome_act/";
    %if isempty(variant)
    %    datapath = join([mainpath, "/distances_act/"],"");
    %    outpath =join([mainpath '/results_distances_act/'],"");
    if strcmp(variant,'plain')
        datapath = join([mainpath, "/conn_plain_act/"],"");
        outpath =join([mainpath '/results_conn_plain_act/'],"");
    elseif strcmp(variant,'fa')
        datapath = join([mainpath, "/famean_act/"],"");
        outpath =join([mainpath '/results_famean_act/'],"");
    end
%else
%    connectomes_folder = "/Volumes/Data/Badea/Lab/mouse/mrtrix_ad_decode/connectome/";
%    datapath = join([mainpath, "/distances/"],"");
%    if isempty(variant)
%        datapath = join([mainpath, "/distances/"],"");
%        outpath =join([mainpath '/results_distances/'],"");%
%
%    elseif variant == 'fa'
%        datapath = join([mainpath, "/famean/"],"");
%        outpath =join([mainpath '/results_famean/'],"");
%    end
end


%outpath = '/Users/alex/Matlab/popNet_HOPCA_ADDecode_2024/results/';
data_source = 2; %1-endstreams, 2-inclusive
trial = '_1'; %today 1st try **important otherwise will overwrite other trials today
%type = 1; %Please pick [1-'All', 2-'noYoung', 3-'onlyYoung']
orth = 1; %Please pick [1-'V_orth', '2-U,V,W_orth']note:we usually orth = 1
gen_type = 1;%Please pick [1-'All', 2-'gen3&4', 3-'only gen0']
k = 15; % # of factors to extract
date = datestr(now,30);

%----------------Main Part of TensorPCA&Variation identify-----------------


if strcmp(analysis_type,'genotype')
    name = 'genotype_comparison';
elseif strcmp(analysis_type,'sex')
    name = 'sex_comparison';
elseif strcmp(analysis_type,'age')
    name = 'age_comparison';
end

% _genotype_4 _genotype_3 
subselect = '';

%load connectivity data
if data_source == 1
    connectomes_path = join([datapath 'connectivity_all' '_ADDecode' '_inclusive_Dipy.mat'],"");
    %connectomes_path = join([mypath 'connectivity_all' '_ADDecode' '_Dipy' subselect variants_str '.mat'],"");
    result_path = [ outpath '/myresults_' name '_inclusive.txt'];
    load(connectomes_path);
elseif data_source == 2
    %connectomes_path = join([datapath 'connectivity' '_ADDecode' '_mrtrix' subselect variant_str '.mat'],"");
    connectomes_path = join([datapath 'connectivity' '_ADDecode' '_mrtrix' subselect '.mat'],"");
    result_path = join([outpath '/myresults_' name subselect variant_str '.txt'],"");
    variance_path = join([outpath '/myvariance_' name subselect variant_str '.txt'],"");
    %response_array_path = join([datapath 'response_array' subselect variant_str '.mat'],"");
    response_array_path = join([datapath 'response_array' subselect '.mat'],"");
    load(connectomes_path);
end
%elseif data_source == 2
%    outpath = [outpath 'Dipy_inclusive/'];
%    connectome_path = join([mypath 'connectivity_all' '_ADDecode' '_inclusive_Dipy.mat'],"");
%    load(connectome_path);
%elseif data_source == 3
%    outpath = [outpath 'Dipy_inclusive/'];
%    connectome_path = join([mypath 'connectivity_all' '_AD_Decode' 'inclusive_Dipy.mat']);
%    load(connectome_path);   
%    connectivity = connectivity_diff;
%end

if ~exist(outpath, 'dir')
   mkdir(outpath)
end

if data_source == 1 || data_source == 2
    
    load(response_array_path);
    response = response_array;
    test = ismember(response,[3 4]);
    
    APOE3 = response(response(:,2)==3, 1);
    APOE4 = response(response(:,2)==4, 1);

    Male = response(response(:,4)==1, 1);
    Female = response(response(:,4)==2, 1);
    
    %connectivity = connectivity(:,:,idx_used(:,1));
    X = tensor(connectivity);
    
    %idx_AMD = (idx_used ==  3 | idx_used == 4);
    %idx = idx_AMD(:,2); 
end

if orth == 1
    [V,D,U,W,Xhat,obj] = hopca_popNet(X,k,foptions);
    orth_type = 'V';
else
    [V,D,U,W,Xhat,obj] = hopca_popNet_new(X,k,foptions);
    orth_type = 'UVW';
end

%Calculate variation percentage
% U subject mode
%V network mode
PCs.U = U;
PCs.V = V;
percent_store = []; %help select # of PCs

filevariance_ID = fopen(variance_path,'w');

for i = 1:k
    [percent] = var_explained(X,i,PCs);
    percent_store = cat(2, percent_store, [i;percent]);
    %txt = (['top ' num2str(i) '/' num2str(k) ' PCs explain ' num2str(percent) '% variance of X' '\n']);
    txt = (['top ' num2str(i) 'out of' num2str(k) ' PCs explain ' num2str(percent) ' percent variance of X' '\n']);
    fprintf(filevariance_ID,txt);
    disp(txt);
end

fclose(filevariance_ID);

%datapath = join([outpath date(3:8) trial '_TNPCA_' orth_type '_' num2str(type) '_K' num2str(k) '_' name '_Outputs.mat'],"");
datapath = join([outpath date(3:8) trial '_TNPCA_' orth_type '_K' num2str(k) '_' name '_Outputs.mat'],"");
save(datapath,'U','V','D','percent_store','obj','connectivity');
%clearvars -except mypath datapath idx_used name idx outpath response_array result_path analysis_type;
load(datapath);

%load([mypath 'response_all.mat']);
%control_idx = control_idx_init_paired;
%AMD_idx = AMD_idx_init_paired;
%idx_control = (idx_init_paired ==  1 | idx_init_paired ==  3 | idx_init_paired ==  4);

%idx_AMD = (idx_used ==  2 | idx_used ==  5 | idx_used == 6);
%idx = idx_AMD(:,2);

if strcmp(analysis_type,'genotype')
    idx = response_array(:,2);
    idx_4 = idx==4;
    idx = idx_4;
elseif strcmp(analysis_type, 'sex')
    idx = response_array(:,4);
    idx_M = idx==1;
    idx = idx_M;    
elseif strcmp(analysis_type,'age')
    idx = response_array(:,3);
    idx_old = idx>50;
    idx = idx_old;  
end

data = cat(2,U(:,1:10),idx);
[w,t,fp]=fisher_training(data(:,1:end-1),data(:,end));
w = w/norm(w);
u0 = mean(data(data(:,end)==0,1:end-1));
u1 = mean(data(data(:,end)==1,1:end-1));
s = norm(u1-u0);

Net_change = zeros(84,84);
for i = 1:size(data(:,1:end-1),2)
    base_net = D(i)*w(i)*(V(:,i)*V(:,i)');
    Net_change = Net_change+base_net;
    %disp(i)
end
Net_change = s*Net_change;

%remove diagnal
Net_change = Net_change-diag(diag(Net_change));

k = 200;
maxcon = maxk(Net_change(:),k);
mincon = mink(Net_change(:),k);
mostcon = cat(1,maxcon,mincon);
mostcon = sort(mostcon,'descend','ComparisonMethod','abs');

%find the ROI index
ridx = [];
cidx = [];
for j = 1:2:k
    disp(mostcon(j))
    [r,c] = find(Net_change==mostcon(j));
    ridx = cat(1,ridx,r);
    cidx = cat(1,cidx,c);
end
mostcon_idx = [ridx cidx];
mostcon_idx = mostcon_idx(1:2:k,:);

%Autonomy_data = readtable('C:\Users\Jacques Stout\Documents\MATLAB\popNet_HOPCA_whiston_2\data\anatomyInfo_whiston_new.csv');
Autonomy_data = readtable('/Users/alex/code/Matlab/popNet_HOPCA_whiston/data/anatomyInfo_whiston_new.csv');

%disp(['the top' num2str(k/2) ' connected subnetwork contribute to distinguish old and young '])
disp(['the top' num2str(k/2) ' connected subnetwork contribute to distinguish control and AMD '])

%disp(' ')

dot_txt_position = strfind(result_path, '.txt');
result_path_char = char(result_path);
result_path_nodes = [result_path_char(1:dot_txt_position(end)-1), '_nodes.txt'];

fileID = fopen(result_path,'w');
%fileID = fopen(['C:\Users\Jacques Stout\Documents\MATLAB\popNet_HOPCA_whiston_2\results/myresults_' name '.txt'],'w');
for h = 1:size(mostcon_idx,1)'
    %h rank of pair
    %con1 Region 1 name
    %d1 - left or right
    %con2
    %d2
    %i1, i2 -index of regions
    i1 = mostcon_idx(h,1);
    i2 = mostcon_idx(h,2);
    con1 = char(Autonomy_data(i1,2).Variables);
    d1 = char(Autonomy_data(i1,3).Variables);
    con2 = char(Autonomy_data(i2,2).Variables);
    d2 = char(Autonomy_data(i2,3).Variables);
    group1_mean = mean(connectivity(i1,i2,idx));
    group2_mean = mean(connectivity(i1,i2,~idx));
    %txt = ([num2str(h) ' ' con1 '_' d1 '---' con2 '_' d2 ' ' num2str(i1) ' ' num2str(i2) ' with weight of ' num2str((maxcon(h*2)/(0.01*s))) ' and avg difference of ' diff_val '\n']);
    txt = ([num2str(h) ' ' con1 '_' d1 '---' con2 '_' d2 ' ' num2str(i1) ' ' num2str(i2) ' with weight of ' num2str((maxcon(h*2)/(0.01*s))) '\n']);
    fprintf(fileID,txt);
    display(txt);
    %disp(' ')
end
fclose(fileID);


if extended_analysis
    for h = 1:size(mostcon_idx,1)'
        i1 = mostcon_idx(h,1);
        i2 = mostcon_idx(h,2);
        APOE4_mean = mean(connectivity(i1,i2,idx));
        APOE3_mean = mean(connectivity(i1,i2,~idx));
        diff_mean = APOE4_mean - APOE3_mean;
    end


    maxcon_evened = maxcon(2:2:end);    
    mostcon_idx_2 = [mostcon_idx, maxcon_evened];

    unique_integers = unique(vertcat(unique(mostcon_idx_2(:, 1), 'rows'), unique(mostcon_idx_2(:, 2), 'rows')));
    num_integers = size(unique_integers);

    integer_weights = zeros(num_integers(1),2);
    i=1;

    for integer=unique_integers'
        row_index = unique(vertcat(find(mostcon_idx_2(:,1) == integer),find(mostcon_idx_2(:,2) == integer)));
        sumweights = sum(mostcon_idx_2(row_index,3));
        integer_weights(i,1)=integer;
        integer_weights(i,2)=sumweights;
        i= i + 1;
        %con = char(Autonomy_data(integer,2).Variables);
        %d = char(Autonomy_data(integer,3).Variables);

        %txt = ([con '_' d ' has total weight of ' num2str((sumweights/(0.01*s))) '\n']);
        %display(txt);
    end

    integer_weights = sortrows(integer_weights, -2);

    fileID = fopen(result_path_nodes,'w');

    for i=1:size(integer_weights,1)
        integer = integer_weights(i,1);
        sumweights = integer_weights(i,2);
        con = char(Autonomy_data(integer,2).Variables);
        d = char(Autonomy_data(integer,3).Variables);

        txt = ([con '_' d ' has total weight of ' num2str((sumweights/(0.01*s))) '\n']);
        fprintf(fileID,txt);

        display(txt);
    end
    fclose(fileID);
end

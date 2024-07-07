library(ggplot2)

library("xlsx")
library("emmeans")
library("ggpubr")
#library('openxlsx')
library('readxl')

path_master = "/Users/jas/jacques/AD_Decode_excels/AD_DECODE_data3.xlsx"
#master = readxl::read_xlsx(path_master)
master = read_excel(path_master)

#master <- subset(master, !(Risk %in% c("MCI", "AD")))
master = master[!is.na(master$genotype),]

geno = master$genotype
geno[master$genotype=="APOE34"] = "APOE44"
geno[master$genotype=="APOE23"] = "APOE33"
master$geno = geno

path_stats= ("/Volumes/Data/Badea/ADdecode.01/Analysis/TractSeg_project/TractSeg_analysis/stats/")
#folder_path = paste0("/Users/jas/jacques/AD_Decode_bundles_figures/bundle_split_results/TractSeg")
tractseg = FALSE
split = TRUE

path_stats= "/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V1_0_reg_insularight_hippocampusright/stats/"
path_stats_excel= "/Volumes/Data/Badea/Lab/AD_Decode/TRK_bundle_splitter/V1_0_reg_insularight_hippocampusright/stats/excel_summary"
output_path = '/Users/jas/jacques/AD_Decode_bundles_figures/bundle_split_results/insularight_hippocampusright_excels'

if (!dir.exists(output_path)) {
  dir.create(output_path)
}
  
edge = TRUE

stat_file_list=list.files(path_stats)

if (tractseg) {
  plain_index = grep("Tractometry_", stat_file_list)
} else {
  plain_index = grep("Tractometry_mrtrixfa", stat_file_list)
}

plain_index = grep("Tractometry_", stat_file_list)
stat_file_list = stat_file_list[plain_index]

all_ids = c('_all')
orig_id <- c('')

list_1 <- c('_0','_1','_2')

combinations_lvl1 <- expand.grid(list_1)
comb_ids_lvl1 <- as.list(apply(combinations_lvl1, 1, paste, collapse = ""))

nosides <- TRUE
combine_lr <- FALSE


bundle_ids_dic <- list()
bundle_ids_dic$`0` <- list()
bundle_ids_dic$`1` <- list()
bundle_ids_dic$`2` <- list()
bundle_ids_dic$`3` <- list()

bundle_ids_dic$`0` <- c(orig_id)
bundle_ids_dic$`1` <- c(comb_ids_lvl1)

master_new <- master

master_new = master_new[!is.na(master_new$genotype),]

master_new$age = as.numeric(master_new$age)

master_new$MRI_Exam <- paste0('S0', master_new$MRI_Exam)
master_new$MRI_Exam <- gsub('S0775', 'S00775', master_new$MRI_Exam)


for (id  in all_ids)
{
  combined_master_df <- master_new
  
  if (combine_lr == TRUE){
    output_path_fig = file.path(output_path,paste0('master_df',id,'_combined.xlsx'))
  } else{
    output_path_fig = file.path(output_path,paste0('master_df',id,'.xlsx'))
  }
  
  num_streamlines_path = file.path(path_stats_excel,paste0("numstreamlines_summary",id,".xlsx"))
  num_sl_data = read_excel(num_streamlines_path)
  
  vol_streamlines_path = file.path(path_stats_excel, paste0("volume_tract_summary",id,".xlsx"))
  vol_sl_data = read_excel(vol_streamlines_path)
  
  if (nosides == FALSE){
    BUAN_path = file.path(path_stats_excel, paste0("BUAN_summary",id,".xlsx"))
    BUAN_data = read_excel(BUAN_path)
  }
  
  len_path = file.path(path_stats_excel,paste0("lenstreamlines_summary",id,".xlsx"))
  len_data = read_excel(len_path)
 
  colnames = colnames(num_sl_data)
  #colnames <- colnames[!names(colnames) %in% "Subj"]
  
  colnames_left = colnames[grep( "left", colnames)]
  colnames_noleftright = gsub ("_left" , "", colnames_left)
  colnames_right = colnames[grep( "right", colnames)]
  
  colnames_left_right <- c(colnames_left,colnames_right)
  
  if (nosides == TRUE){
    colnames_numsl = paste(colnames_noleftright, "_num_sl", sep = "")
    colnames_volsl = paste(colnames_noleftright, "_vol_sl", sep = "")
    colnames_lensl = paste(colnames_noleftright, "_len_sl", sep = "")
  }
  else {
    colnames_numsl = paste(colnames_left_right, "_num_sl", sep = "")
    colnames_volsl = paste(colnames_left_right, "_vol_sl", sep = "")
    colnames_lensl = paste(colnames_left_right, "_len_sl", sep = "")
    colnames_BUAN = paste(colnames_noleftright, "_BUAN", sep = "")
  }
  
  colnames_numsl_ass = paste(colnames_noleftright, "_num_sl_assym", sep = "")
  colnames_volsl_ass = paste(colnames_noleftright, "_vol_sl_assym", sep = "")
  colnames_lensl_ass = paste(colnames_noleftright, "_len_sl_assym", sep = "")
  
  names(num_sl_data) <- c('Subj',colnames_numsl)
  names(vol_sl_data) <- c('Subj',colnames_volsl)
  names(len_data) <- c('Subj',colnames_lensl)
  
  combined_master_df <- merge(combined_master_df, num_sl_data, by.x = "MRI_Exam", by.y ="Subj", all.x = TRUE)
  combined_master_df <- merge(combined_master_df, vol_sl_data, by.x = "MRI_Exam", by.y ="Subj", all.x = TRUE)
  combined_master_df <- merge(combined_master_df, len_data, by.x = "MRI_Exam", by.y ="Subj", all.x = TRUE)
  
  if (nosides==FALSE){
    names(BUAN_data) <- c('Subj',colnames_BUAN)
    combined_master_df <- merge(combined_master_df, BUAN_data, by.x = "MRI_Exam", by.y ="Subj", all.x = TRUE)
    combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_numsl_ass, function(x) x=NA), colnames_numsl_ass) )
    combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_volsl_ass, function(x) x=NA), colnames_volsl_ass) )
    combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_lensl_ass, function(x) x=NA), colnames_lensl_ass) )
  }
  
  combined_master_df <- combined_master_df[complete.cases(combined_master_df[, 49]), ]
  
  if (nosides == FALSE){
    for (j in colnames_numsl_ass) {
      colname_base = gsub ("_assym" , "", j)
      L = combined_master_df[,gsub("bundle_", "bundle_left_", colname_base)]
      R = combined_master_df[,gsub("bundle_", "bundle_right_", colname_base)]
      combined_master_df[,j] = ( abs(L - R) / (L+R ) )
    }
    
    for (j in colnames_volsl_ass) {
      colname_base = gsub ("_assym" , "", j)
      L = combined_master_df[,gsub("bundle_", "bundle_left_", colname_base)]
      R = combined_master_df[,gsub("bundle_", "bundle_right_", colname_base)]
      combined_master_df[,j] = ( abs(L - R) / (L+R ) )
    }
    
    for (j in colnames_lensl_ass) {
      colname_base = gsub ("_assym" , "", j)
      L = combined_master_df[,gsub("bundle_", "bundle_left_", colname_base)]
      R = combined_master_df[,gsub("bundle_", "bundle_right_", colname_base)]
      combined_master_df[,j] = ( abs(L - R) / (L+R ) )
    }
  }
  
  depth <- as.character(length(gregexpr("_all", id)[[1]])-1)
  
  bundle_ids <- bundle_ids_dic[[depth]]
  
  for (bundle_id in bundle_ids){
    if (bundle_id == ''){
      pattern <- "_[0-9]\\.csv$"
      stat_file_list_id <- stat_file_list[!grepl(pattern, stat_file_list)]
    } else {
      pattern <- paste0("S[0-9][0-9][0-9][0-9][0-9]",bundle_id,'.csv')
      stat_file_list_id <- stat_file_list[grepl(pattern, stat_file_list)]
    }
    
    if (tractseg) {
      temp_data = read.csv(paste0(path_stats,stat_file_list_id[1]), sep = ";")
    } else {
      temp_data = read.csv(paste0(path_stats,stat_file_list_id[1]), sep = ",")
    }
    colnames = colnames(temp_data)
    
    temp_data = temp_data [,2:dim(temp_data)[2],drop=FALSE]
    colnames = colnames[-1]
    
    master = as.data.frame(master)
    
    colnames_meanfa = paste(colnames, "_meanfa", sep = "")
    colnames_sdfa = paste(colnames, "_sdfa", sep = "")
    
    colnames_left = colnames[grep( "left", colnames)]
    if (nosides==TRUE){
      colnames_noleftright = colnames[grep( "bundle", colnames)]
    }
    else{
      colnames_noleftright = gsub ("_left" , "", colnames_left)
    }
    
    colnames_meanass = paste(colnames_noleftright, "_meanfa_assym", sep = "")
    colnames_sdass = paste(colnames_noleftright, "_sdfa_assym", sep = "")
    
    combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_meanfa, function(x) x=NA), colnames_meanfa) )
    combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_sdfa, function(x) x=NA), colnames_sdfa) )
    
    if (nosides == FALSE){
      combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_meanass, function(x) x=NA), colnames_meanass) )
      combined_master_df = cbind(combined_master_df, setNames( lapply(colnames_sdass, function(x) x=NA), colnames_sdass) )
    }
    
    bundle_sub <- ''
    
    bundle_sub <- paste0(bundle_sub,'.csv')
    
    for (i in 1:dim( combined_master_df)[1]){
      
      index_master <- c()
      #index_master =  which( combined_master_df$MRI_Exam[i]  == as.numeric(substr(stat_file_list_id,23,27)) )
      
      if (is.na( combined_master_df$MRI_Exam[i] )){
        break
      }
      
      for (j in seq_along(stat_file_list_id)) {
        subj_pattern <- paste0("_S[0-9]")
        match <- regexpr(subj_pattern, stat_file_list_id[j])
        if (match==-1){
          next
        }
        subj_num = substr(stat_file_list_id[j], match + 3, match + 6)
        if ( substring(combined_master_df$MRI_Exam[i],3) == subj_num) {
          if (endsWith(stat_file_list_id[j], bundle_sub)) {
            index_master <- j
          }
        }
      }
      
      if (length(index_master) > 0){
        if (tractseg) {
          data = read.csv(paste0(path_stats,stat_file_list_id[index_master]), sep = ";")
        } else {
          data = read.csv(paste0(path_stats,stat_file_list_id[index_master]), sep = ",")
        }
        
        data = data[,2:dim(data)[2],drop=FALSE]
        ####
        # dim(data)
        # c= c(seq(25:75))
        # data = data[c, ]
        ####
        combined_master_df[i,colnames_meanfa] = sapply(data, mean)
        combined_master_df[i,colnames_sdfa] = sapply(data, sd)
        # combined_master_df[i,colnames_ass] = sapply(data, mean)
        
        if (nosides == FALSE){
          for (j in colnames_meanass) {
            colname_base = gsub ("_meanfa_assym" , "", j)
            if (tractseg){
              L = data[,paste0(colname_base, '_left')]
              R = data[,paste0(colname_base, '_right')]
            } else{
              L = data[,gsub("bundle_", "bundle_left_", colname_base)]
              R = data[,gsub("bundle_", "bundle_right_", colname_base)]
            }
            combined_master_df[i,j] = mean( abs(L - R) / (L+R ) )*200
          }
          
          for (j in colnames_sdass) {
            colname_base = gsub ("_sdfa_assym" , "", j)
            if (tractseg){
              L = sd(data[,paste0(colname_base, '_left')])
              R = sd(data[,paste0(colname_base, '_right')])
            } else{
              L = sd(data[,gsub("bundle_", "bundle_left_", colname_base)])
              R = sd(data[,gsub("bundle_", "bundle_right_", colname_base)])
            }
            combined_master_df[i,j] = mean( abs(L - R) / (L+R ) )
          }
        }
      }
    }
  }
  
  if (combine_lr == TRUE){
    colnames = colnames(combined_master_df)
    colnames_left = colnames[grep( "left", colnames)]
    colnames_noleftright = gsub ("_left" , "", colnames_left)
    colnames_right = colnames[grep( "right", colnames)]
    
    for (colname_l in colnames_left){
      colname_r <- gsub("bundle_left", "bundle_right", colname_l)
      colname_lr <- gsub("bundle_left", "bundle", colname_l)
      L = combined_master_df[,colname_l]
      R = combined_master_df[,colname_r]
      
      if (grepl('len_sl', colname_lr) || (grepl('mean_fa', colname_lr)) || (grepl('sd_fa', colname_lr))){
        combined_master_df[,colname_l] = (L+R)/2
      } else{
        combined_master_df[,colname_l] = L+R
      }
      combined_master_df <- combined_master_df[, !names(combined_master_df) %in% colname_r]
      names(combined_master_df)[names(combined_master_df) == colname_l] <- colname_lr
    }
  }
  
  combined_master_df = combined_master_df[!is.na(combined_master_df$genotype),]
  write.xlsx(combined_master_df, output_path_fig)
}



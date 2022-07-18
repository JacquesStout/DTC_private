#read csv file
#plot FA Along tracts for group 1
#plot FA along tracts for group 2
# stats
library(ggplot2)
library(readxl)
library(lme4)
library(visreg)
library(tidyr)
library(magrittr) # need to run every time you start R and want to use %>%
library(dplyr)
library(Rmisc)

project = 'AMD'
ratio = 1

std_mean <- function(x) sd(x)/sqrt(length(x))

path_join <- function(path){
  if (Sys.info()['sysname']=='Windows') {
    newpath <- paste(unlist(path), collapse='\\')
  }
  else{
    newpath <- paste(unlist(path), collapse='/')
  }
  return(newpath)
}

my.file.rename <- function(from, to) {
  todir <- dirname(to)
  if (!isTRUE(file.info(todir)$isdir)) dir.create(todir, recursive=TRUE)
  file.rename(from = from,  to = to)
}


if (Sys.info()[['user']]=='JacquesStout') {
  atlas_file='C:\\Users\\Jacques Stout\\Documents\\Work\\atlases\\IITmean_RPI_index.xlsx'
  if (project == 'AMD') {
    mainpath = 'C:\\Users\\Jacques\ Stout\\Documents\\Work\\AMD\\'
    outputpath= 'C:\\Users\\Jacques\ Stout\\Documents\\Work\\AMD\\R_figures\\'
    
  }
}
if (Sys.info()[['user']]=='jas') {
  atlas_file='/Volumes/Data/Badea/Lab/atlases/IITmean_RPI/IITmean_RPI_index.xlsx'
  if (project=='AD_Decode') {
    mainpath = '/Volumes/Data/Badea/Lab/human/AD_Decode/Analysis/'
    outputpath='/Users/jas/jacques/AD_decode_abstract/R_figures/'
  }
  if (project == 'AMD') {
    mainpath = '/Volumes/Data/Badea/Lab/human/AMD/'
    outputpath= '/Users/jas/jacques/Whiston_article/R_figures/'
  }
}

if (ratio == 1) {ratio_str = 'all'} else {ratio_str = paste('ratio_',as.character(ratio),sep='')}
if (ratio == 1) {folder_str = ''} else {folder_str = paste('_',as.character(ratio))}

space_param = '_affinerigid'

my_atlas <- read_excel(atlas_file)
outputpath_singlebundle = 
  csvpath_folder = path_join(list(mainpath, paste('Statistics',space_param,'_non_inclusive_symmetric',as.character(folder_str),sep='')))

myYear = 'Initial'
if (myYear=='Initial'){
  groups =list('Paired Initial AMD','Paired Initial Control')
}
if (myYear=='2Year'){
  groups =list('Paired 2-YR AMD','Paired 2-YR Control')
}
#groups =list('Paired Initial AMD','Paired Initial Control')
#groups =list('1-Control','2-AMD')
group_colors <- list('blueviolet','chartreuse1')

regions_all <- vector(mode = 'list', length = 8)

cutoff=10

#regions_all <- list(list(28, 9),list(62, 1),list(77, 43),list(61, 29))
regions_all <- list(list(62, 28),list(58, 45),list(28, 9),list(62, 1),list(77, 43),list(61, 29))
#references <- list(fa,md)
overwrite=TRUE

for (z in 1:6) {
  regions = regions_all[[z]]
  regions_str = tolower(paste(my_atlas[regions[[1]],3],'_',my_atlas[regions[[1]],4],'_to_',my_atlas[regions[[2]],3],'_',my_atlas[regions[[2]],4],sep=''))
  
  g3file = path_join(list(csvpath_folder,paste(groups[1],'_',regions_str,'_',ratio_str,'_bundle_stats.csv',sep='')))
  g4file = path_join(list(csvpath_folder,paste(groups[2],'_',regions_str,'_',ratio_str,'_bundle_stats.csv',sep='')))
  
  g3<-read.csv(g3file, header = TRUE)
  g4<-read.csv(g4file, header = TRUE)
  
  df3<-data.frame(g3)
  df3<-cbind(df3,genotype=groups[[1]])
  
  df4<-data.frame(g4)
  df4<-cbind(df4,genotype=groups[[2]])
  
  df<-rbind(df3,df4)
  
  #these csv files only contain the top 6 clusters
  #see how big are those clusters
  clus3bundleid<-unique(g3$Bundle.ID)
  clus4bundleid<-unique(g4$Bundle.ID)
  
  length(clus3bundleid)
  
  vec <- vector()
  bundlesize3 <- c(vec, length(clus3bundleid))
  bundlesize4<-bundlesize3 
  
  
  for (i in 1:length(clus3bundleid)){
    bundlesize3[i]<-length(which(g3$Bundle.ID==clus3bundleid[i]))
    bundlesize4[i]<-length(which(g4$Bundle.ID==clus4bundleid[i]))
  }
  
  clus3bundleid<-unique(g3$Bundle.ID)
  #plot signal along fibers
  
  dfstream3Bundlei<-df[df$genotype==groups[[1]] & is.element(df$Bundle.ID, clus3bundleid[1:cutoff]),]
  dfstream4Bundlei<-df[df$genotype==groups[[2]] & is.element(df$Bundle.ID, clus4bundleid[1:cutoff]),]
  dfstreamBundlei<-rbind(dfstream3Bundlei,dfstream4Bundlei)
  dfstreamBundlei$genotypenorm <- ((-1)*as.numeric(dfstreamBundlei$genotype) - 3.5)
  
  dfstreamBundlei$lwr=dfstreamBundlei$fa-1.96*STDERR(dfstreamBundlei$fa)
  dfstreamBundlei$hi=dfstreamBundlei$fa+1.96*STDERR(dfstreamBundlei$fa)
  
  #dfstream3Bundlemeans <- dfstream3Bundlei[!duplicated(df$Streamlines.ID), ]
  #dfstream3Bundlemeans <- dfstream3Bundlemeans %>% drop_na(fa)
  #dfstream4Bundlemeans <- dfstream4Bundlei[!duplicated(df$Streamlines.ID), ]
  #dfstream4Bundlemeans <- dfstream4Bundlemeans %>% drop_na(fa)
  #dfstreamBundleameans <- rbind(dfstream3Bundlemeans,dfstream4Bundlemeans)
   
  dfstreamBundleameans <- dfstreamBundlei[!duplicated(df$Streamlines.ID), ]
  dfstreamBundleameans <- dfstreamBundleameans %>% drop_na(fa)
  
  for (streamlineID in unique(dfstreamBundlei$Streamlines.ID)){
    dfstreamBundleameans['Local.coherence'][dfstreamBundleameans['Streamlines.ID'] == streamlineID] = mean(dfstreamBundlei['Local.coherence'][dfstreamBundlei['Streamlines.ID']==streamlineID])
    dfstreamBundleameans['fa'][dfstreamBundleameans['Streamlines.ID'] == streamlineID] = mean(dfstreamBundlei['fa'][dfstreamBundlei['Streamlines.ID']==streamlineID])
    dfstreamBundleameans['md'][dfstreamBundleameans['Streamlines.ID'] == streamlineID] = mean(dfstreamBundlei['md'][dfstreamBundlei['Streamlines.ID']==streamlineID])
    dfstreamBundleameans['ad'][dfstreamBundleameans['Streamlines.ID'] == streamlineID] = mean(dfstreamBundlei['ad'][dfstreamBundlei['Streamlines.ID']==streamlineID])
    dfstreamBundleameans['rd'][dfstreamBundleameans['Streamlines.ID'] == streamlineID] = mean(dfstreamBundlei['rd'][dfstreamBundlei['Streamlines.ID']==streamlineID])
  }
  #%start save stats per bundle: FA and Length
  
  mytLength<-t.test(Length ~ genotype, data = dfstreamBundleameans)
  mytCoherence<-t.test(Streamline.coherence ~ genotype, data = dfstreamBundleameans)
  mytFA<-t.test(fa ~ genotype, data = dfstreamBundleameans)
  mytMD<-t.test(md ~ genotype, data = dfstreamBundleameans)
  mytAD<-t.test(ad ~ genotype, data = dfstreamBundleameans)
  mytRD<-t.test(rd ~ genotype, data = dfstreamBundleameans)
  
  #mytFA<-t.test(fa ~ genotype, data = dfstreamBundlei)
  #mytMD<-t.test(md ~ genotype, data = dfstreamBundlei)
  #mytAD<-t.test(ad ~ genotype, data = dfstreamBundlei)
  #mytRD<-t.test(rd ~ genotype, data = dfstreamBundlei)
  
  size_bundle <- list(length(unique(dfstream3Bundlei$Streamlines.ID)),length(unique(dfstream4Bundlei$Streamlines.ID)))
  #size_bundle_1 <- length(unique(dftemp3$Streamlines.ID))
  #size_bundle_2 <- length(unique(dftemp4$Streamlines.ID)) 
  groups_order <- list(2,1)
  
  groups[as.numeric(groups_order[1])]
  
  references <- c('Length', 'Local.coherence', 'fa', 'md', 'rd', 'ad')
  meanframes = {}
  
  #meansLength_1 = dfstreamBundleameans['Length'][dfstreamBundleameans['genotype']==groups[as.numeric(groups_order[1])]]

  boundaries = {}
  cohen = {}
  for (ref in references){
    #paste('means','Length','_1',sep='')
    df1 = dfstreamBundleameans[ref][dfstreamBundleameans['genotype']==groups[as.numeric(groups_order[1])]]
    boundaries[paste(ref,groups[as.numeric(groups_order[1])],'low')] = mean(df1) - ((1.96 * mean(df1)/sqrt(length(df1))))
    boundaries[paste(ref,groups[as.numeric(groups_order[1])],'high')] = mean(df1) + ((1.96 * mean(df1)/sqrt(length(df1))))
    
    df2 = dfstreamBundleameans[ref][dfstreamBundleameans['genotype']==groups[as.numeric(groups_order[2])]]
    boundaries[paste(ref,groups[as.numeric(groups_order[2])],'low')] = mean(df2) - ((1.96 * mean(df2)/sqrt(length(df2))))
    boundaries[paste(ref,groups[as.numeric(groups_order[2])],'high')] = mean(df2) + ((1.96 * mean(df2)/sqrt(length(df2))))
    
    cohen[ref] = (mean(df1) - mean(df2)) / sqrt((var(df1)+var(df2))/2)

  }
  
  mytableLength<-as_data_frame(
    cbind(mytLength$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytLength$estimate[as.numeric(groups_order[1])], boundaries[paste('Length',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('Length',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytLength$estimate[as.numeric(groups_order[2])],boundaries[paste('Length',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('Length',groups[as.numeric(groups_order[2])],'low')],cohen['Length'],mytLength$statistic, mytLength$p.value,  
          mytLength$conf.int[1], mytLength$conf.int[2] , mytLength$parameter)
  )
  
  mytableCoherence<-as_data_frame(
    cbind(mytCoherence$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytCoherence$estimate[as.numeric(groups_order[1])], boundaries[paste('Local.coherence',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('Local.coherence',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytCoherence$estimate[as.numeric(groups_order[2])],  boundaries[paste('Local.coherence',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('Local.coherence',groups[as.numeric(groups_order[2])],'low')],cohen['Local.coherence'],mytCoherence$statistic, mytCoherence$p.value,  
          mytCoherence$conf.int[1], mytCoherence$conf.int[2] , mytCoherence$parameter)
  )
  
  mytableFA<-as_data_frame(
    cbind(mytFA$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytFA$estimate[as.numeric(groups_order[1])], boundaries[paste('fa',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('fa',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytFA$estimate[as.numeric(groups_order[2])],  boundaries[paste('fa',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('fa',groups[as.numeric(groups_order[2])],'low')],cohen['fa'],mytFA$statistic, mytFA$p.value,  
          mytFA$conf.int[1], mytFA$conf.int[2] , mytFA$parameter)
  )

  mytableMD<-as_data_frame(
    cbind(mytMD$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytMD$estimate[as.numeric(groups_order[1])], boundaries[paste('md',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('md',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytMD$estimate[as.numeric(groups_order[2])],  boundaries[paste('md',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('md',groups[as.numeric(groups_order[2])],'low')],cohen['md'],mytMD$statistic, mytMD$p.value,  
          mytMD$conf.int[1], mytMD$conf.int[2] , mytMD$parameter)
  )
  
  mytableAD<-as_data_frame(
    cbind(mytAD$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytAD$estimate[as.numeric(groups_order[1])], boundaries[paste('ad',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('ad',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytAD$estimate[as.numeric(groups_order[2])],  boundaries[paste('ad',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('ad',groups[as.numeric(groups_order[2])],'low')],cohen['ad'],mytAD$statistic, mytAD$p.value,  
          mytAD$conf.int[1], mytAD$conf.int[2] , mytAD$parameter)
  )

  mytableRD<-as_data_frame(
    cbind(mytRD$data.name, as.numeric(size_bundle[as.numeric(groups_order[1])]), mytRD$estimate[as.numeric(groups_order[1])], boundaries[paste('rd',groups[as.numeric(groups_order[1])],'high')],boundaries[paste('rd',groups[as.numeric(groups_order[1])],'low')], 
          as.numeric(size_bundle[as.numeric(groups_order[2])]), mytRD$estimate[as.numeric(groups_order[2])], boundaries[paste('rd',groups[as.numeric(groups_order[2])],'high')],boundaries[paste('rd',groups[as.numeric(groups_order[2])],'low')],cohen['rd'],mytRD$statistic, mytRD$p.value,  
          mytRD$conf.int[1], mytRD$conf.int[2] , mytRD$parameter)
  )
  
  # col.names=c('contrast', names(mytFA$estimate)[1],names(mytFA$estimate)[2],names(mytFA$statistic),'pvalue', 'CIlwr','CIhi', 'df'))
  mycolnames<-c('contrast',paste('num str ',groups[as.numeric(groups_order[1])],sep=''), names(mytFA$estimate)[as.numeric(groups_order[1])], paste('High bound ',groups[as.numeric(groups_order[1])],sep=''), paste('Low bound ',groups[as.numeric(groups_order[1])],sep=''), paste('num str ',groups[as.numeric(groups_order[2])],sep=''), names(mytFA$estimate)[as.numeric(groups_order[2])],  paste('High bound ',groups[as.numeric(groups_order[2])],sep=''), paste('Low bound ',groups[as.numeric(groups_order[2])],sep=''), 'Cohen', names(mytFA$statistic), 'pvalue', 'CIlwr', 'CIhi', 'df')
  
  myfile<-paste(outputpath,myYear,'_',regions_str,'_bundles_all_stats_cutoff_',cutoff,'.csv',sep='')
  
  if (!file.exists(myfile) || overwrite==TRUE){
    #add column names for the first instance
    if (file.exists(myfile) & overwrite==TRUE){
      file.remove(myfile)
    }
    write.table(mytableLength, file=myfile, col.names = mycolnames , sep = ',' , row.names = F,append=TRUE)
    #remove columnnames for subsequent rows
    write.table(mytableCoherence, file=myfile, sep = ',' , row.names = F,append=TRUE,col.names=FALSE)
    write.table(mytableFA, file=myfile, sep = ',' , row.names = F,append=TRUE,col.names=FALSE)
    write.table(mytableMD, file=myfile, sep = ',' , row.names = F,append=TRUE,col.names=FALSE)
    write.table(mytableAD, file=myfile, sep = ',' , row.names = F,append=TRUE,col.names=FALSE)
    write.table(mytableRD, file=myfile, sep = ',' , row.names = F,append=TRUE,col.names=FALSE)
  }
}
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
  atlas_file='C:\\Users\\JacquesStout\\Documents\\Work\\atlases\\IITmean_RPI_index.xlsx'
  if (project == 'AMD') {
    mainpath = 'C:\\Users\\JacquesStout\\Documents\\Work\\AMD'
    outputpath= 'C:\\Users\\JacquesStout\\Documents\\Work\\AMD\\R_figures'
    
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
    outputpath= '/Users/jas/jacques/Whiston_article/R_figures_confidence_test/'
  }
}

if (ratio == 1) {ratio_str = 'all'} else {ratio_str = paste('ratio_',as.character(ratio),sep='')}
if (ratio == 1) {folder_str = ''} else {folder_str = paste('_',as.character(ratio))}

my_atlas <- read_excel(atlas_file)
outputpath_singlebundle = 
  csvpath_folder = path_join(list(mainpath, paste('Statistics_MDT_non_inclusive_symmetric',as.character(folder_str),sep='')))

#(9, 1),(24, 1), (76, 42), (76, 64), (77, 9), (43, 9)
groups =list('Paired_Initial_Control_','Paired_Initial_AMD_')
groups =list('Paired Initial Control_','Paired Initial AMD_')
groups =list('Paired 2-YR Control_','Paired 2-YR AMD_')
groups_str =list('Paired Initial Control','Paired Initial AMD')
groups_str =list('Paired 2-YR Control','Paired 2-YR AMD')

regions_all <- vector(mode = 'list', length = 10)
regions_all[[1]] <- list(9, 1)
regions_all[[2]] <- list(24, 1)
regions_all[[3]] <- list (76, 42)
regions_all[[4]] <- list(76, 64)
regions_all[[5]] <- list(77, 9)

regions_all <- list(list(9, 1),                   # Create nested list using list()
                    list(24, 1),
                    list (76, 42), list(76, 64), list(77, 9))

for (z in 1:5) {
  regions = regions_all[[z]]
  regions_str = tolower(paste(my_atlas[regions[[1]],3],'_',my_atlas[regions[[1]],4],'_to_',my_atlas[regions[[2]],3],'_',my_atlas[regions[[2]],4],sep=''))
  
  
  g3file = path_join(list(csvpath_folder,paste(groups[1],regions_str,'_',ratio_str,'_bundle_stats_fa.csv',sep='')))
  g4file = path_join(list(csvpath_folder,paste(groups[2],regions_str,'_',ratio_str,'_bundle_stats_fa.csv',sep='')))
  
  g3<-read.csv(g3file, header = TRUE)
  g4<-read.csv(g4file, header = TRUE)
  
  df3<-data.frame(g3)
  df3<-cbind(df3,genotype=groups_str[[1]])
  
  df4<-data.frame(g4)
  df4<-cbind(df4,genotype=groups_str[[2]])
  
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
  
  # 
  #   ggplot(df, aes(x=fa, color=genotype, fill=genotype)) +
  #     geom_histogram( alpha=0.5, position='identity')+
  #     geom_density(alpha=0.6)+
  #     theme_classic()+
  #     scale_color_manual(values=c('chartreuse1','blueviolet'))+
  #     scale_fill_manual(values=c('chartreuse1','blueviolet'))+
  #     theme(legend.position='top')+
  #     theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
  #           axis.text.y = element_text(face='bold', size=14, angle=0),
  #           axis.line.x = element_line(colour = 'black', size=0.5),
  #           axis.line.y = element_line(colour = 'black', size=0.5),
  #           # panel.grid.major = element_blank(),
  #           panel.grid.minor = element_blank(),
  #           # panel.border = element_blank(),
  #           panel.background = element_blank())+
  #     ggtitle(regions_str)
  # 
  #   ggsave(path_join(list(outputpath,paste(regions_str,'_allbundles_FAHist','_',ratio_str,'.pdf',sep=''))), plot = last_plot(), device = 'pdf',
  #          scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
  # 
  #   dftemp3<-subset(df, (genotype==groups_str[[1]] & Bundle.ID==clus3bundleid[i]))
  #   dftemp4<-subset(df, (genotype==groups_str[[2]] & Bundle.ID==clus4bundleid[i]))
  # 
  #   ggplot(df, aes(x=Length, color=genotype, fill=genotype)) +
  #     geom_histogram( alpha=0.5, position='identity')+
  #     geom_density(alpha=0.6)+
  #     theme_classic()+
  #     scale_color_manual(values=c('chartreuse1','blueviolet'))+
  #     scale_fill_manual(values=c('chartreuse1','blueviolet'))+
  #     theme(legend.position='top')+
  #     theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
  #           axis.text.y = element_text(face='bold', size=14, angle=0),
  #           axis.line.x = element_line(colour = 'black', size=0.5),
  #           axis.line.y = element_line(colour = 'black', size=0.5),
  #           # panel.grid.major = element_blank(),
  #           panel.grid.minor = element_blank(),
  #           # panel.border = element_blank(),
  #           panel.background = element_blank())
  # 
  #   ggsave(path_join(list(outputpath,paste(regions_str,'_allbundles_LengthHist','_',ratio_str,'.pdf',sep=''))), plot = last_plot(), device = 'pdf',
  #          scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
  # 
  # 
  # 
  #   #darkorchid2
  #   #geom_histogram( fill='white', alpha=0.5, position='identity')+
  #   #geom_histogram( alpha=0.5, position='identity')+
  # 
  #   for (i in 1:length(clus3bundleid)){
  #     dftemp3<-subset(df, (genotype==groups_str[[1]] & Bundle.ID==clus3bundleid[i]))
  #     dftemp4<-subset(df, (genotype==groups_str[[2]] & Bundle.ID==clus4bundleid[i]))
  #     dftemp<-rbind(dftemp3,dftemp4)
  # 
  #     ggplot(dftemp, aes(x=fa, color=genotype, fill=genotype)) +
  #       geom_histogram( alpha=0.5, position='identity')+
  #       geom_density(alpha=0.6)+
  #       ggtitle(paste('Bundle/Cluster', toString(i), sep= ' '))+
  #       theme_classic()+
  #       scale_color_manual(values=c('chartreuse1','blueviolet'))+
  #       scale_fill_manual(values=c('chartreuse1','blueviolet'))+
  #       theme(legend.position='top')+
  #       theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
  #             axis.text.y = element_text(face='bold', size=14, angle=0),
  #             axis.line.x = element_line(colour = 'black', size=0.5),
  #             axis.line.y = element_line(colour = 'black', size=0.5),
  #             # panel.grid.major = element_blank(),
  #             panel.grid.minor = element_blank(),
  #             # panel.border = element_blank(),
  #             panel.background = element_blank())
  # 
  #     ggsave(path_join(list(outputpath,paste(regions_str,'_bundle', toString(i), '_FAHist','_',ratio_str,'.pdf',sep=''))), plot = last_plot(), device = 'pdf',
  #            scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
  #   }
  # 
  #   for (i in 1:length(clus3bundleid)){
  #     dftemp3<-subset(df, (genotype==groups_str[[1]] & Bundle.ID==clus3bundleid[i]))
  #     dftemp4<-subset(df, (genotype==groups_str[[2]] & Bundle.ID==clus4bundleid[i]))
  #     dftemp<-rbind(dftemp3,dftemp4)
  #     #print(i)
  #     #if (i == 10)
  #     #  {ratio_str = 'all'}
  # 
  #     ggplot(dftemp, aes(x=Length, color=genotype, fill=genotype)) +
  #       geom_histogram( alpha=0.5, position='identity')+
  #       geom_density(alpha=0.6)+
  #       ggtitle(paste('Bundle/Cluster', toString(i), sep= ' '))+
  #       theme_classic()+
  #       scale_color_manual(values=c('chartreuse1','blueviolet'))+
  #       scale_fill_manual(values=c('chartreuse1','blueviolet'))+
  #       theme(legend.position='top')+
  #       theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
  #             axis.text.y = element_text(face='bold', size=14, angle=0),
  #             axis.line.x = element_line(colour = 'black', size=0.5),
  #             axis.line.y = element_line(colour = 'black', size=0.5),
  #             # panel.grid.major = element_blank(),
  #             panel.grid.minor = element_blank(),
  #             # panel.border = element_blank(),
  #             panel.background = element_blank())
  # 
  #     ggsave(path_join(list(outputpath,paste(regions_str,'_bundle', toString(i), '_LengthHist','_',ratio_str,'.pdf',sep=''))), plot = last_plot(), device = 'pdf',
  #            scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
  #   }
  
  
  #write these results to tables
  for (i in 1:length(clus3bundleid)){
    i <- 5
    dftemp3<-subset(df, (genotype==groups_str[[1]] & Bundle.ID==clus3bundleid[i]))
    dftemp4<-subset(df, (genotype==groups_str[[2]] & Bundle.ID==clus4bundleid[i]))
    dftemp<-rbind(dftemp3,dftemp4)
    ksbundlefa<-ks.test(dftemp3$fa, dftemp4$fa, alternative = c('two.sided'),exact = NULL)
    ksbundleLength<-ks.test(dftemp3$Length, dftemp4$Length, alternative = c('two.sided'),exact = NULL)
    
    #plot signal along fibers
    
    dfstream3Bundlei<-subset(df, (genotype==groups_str[[1]] & Bundle.ID==clus3bundleid[i]))
    dfstream4Bundlei<-subset(df, (genotype==groups_str[[2]] & Bundle.ID==clus4bundleid[i]))
    dfstreamBundlei<-rbind(dfstream3Bundlei,dfstream4Bundlei)
    dfstreamBundlei$genotypenorm <- ((-1)*as.numeric(dfstreamBundlei$genotype) - 3.5)
    
    #this models does not work very well, but let's explore and make it better later
    #http://bbolker.github.io/mixedmodels-misc/glmmFAQ.html#is-the-likelihood-ratio-test-reliable-for-mixed-models
    fm1 <- lmer(
      formula = fa ~ poly(Point.ID,degree=4)+genotype+ (Point.ID|Streamlines.ID), data = dfstreamBundlei
      #+ , 
    )
    cV <- ranef(fm1, condVar = TRUE)   
    ranvar <- attr(cV[[1]], 'postVar')
    
    
    
    
    
    #dfstreamBundlei$lwr=dfstreamBundlei$fa-1.96*STDERR(dfstreamBundlei$fa)
    #dfstreamBundlei$hi=dfstreamBundlei$fa+1.96*STDERR(dfstreamBundlei$fa)
    
    ggplot(dfstreamBundlei, aes(y=fa, x=Point.ID, color=genotype)) +
      #geom_smooth(stat = 'summary', group=genotype, fun.data = function(y) data.frame(ymin = quantile(y, .05), y = mean(y), ymax = quantile(y, .95))) +
      #geom_smooth(stat = 'summary', fun.data = function(y) data.frame(ymin = quantile(y, .025), y = mean(y), ymax = quantile(y, .975))) +
      #geom_smooth(stat = 'summary', fun.data = mean_cl_quantile)
      #stat_smooth() +
      geom_smooth(data=dfstreamBundlei, se = TRUE, level=0.95, aes(color=genotype,fill=genotype))+
      ##bad crashes geom_smooth(aes(ymin = lwr, ymax = hi,color=genotype),stat = 'identity')+
      ##bad crashes geom_ribbon(data=dfstreamBundlei,aes(ymin=lwr,ymax=hi),alpha=0.3)+
      #aes(ymin=lwr,ymax=hi,fill=genotype, colour=genotype),stat = 'identity')+
      #geom_histogram( alpha=0.5, position='identity')+
      #geom_density(alpha=0.6)+
      ggtitle(paste('FA along streamlines', toString(i), sep= ' '))+
      theme_classic()+
      scale_color_manual(values=c('chartreuse1','blueviolet'))+
      scale_fill_manual(values=c('chartreuse1','blueviolet'))+
      theme(legend.position='top')+
      theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
            axis.text.y = element_text(face='bold', size=14, angle=0),
            axis.line.x = element_line(colour = 'black', size=0.5),
            axis.line.y = element_line(colour = 'black', size=0.5),
            # panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            # panel.border = element_blank(),
            panel.background = element_blank())
    ggsave(path_join(list(outputpath,paste(regions_str,'_alongbundleCI95', toString(i), '_FA_nolwrhi.pdf',sep=''))), plot = last_plot(), device = 'pdf',
           scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
    
    
    
    
    
    
    
    
    
    fapred <- predict(fm1,data=dfstreamBundlei,re.form=NA)
    fapred <- predict(fm1,data=dfstreamBundlei,interval='confidence', level=0.95)
    
    ggplot(dfstreamBundlei, aes(y=fapred, x=Point.ID, color=genotype)) +
      #geom_point()+
      geom_smooth(data=fapred, se = TRUE, level=0.95, aes(color=genotype,fill=genotype))+
      #geom_line(aes(group = genotype), alpha = .3) +
      #geom_line(data = dfstreamBundlei, alpha = .8, size = 1) +
      stat_smooth() +
      #geom_histogram( alpha=0.5, position='identity')+
      #geom_density(alpha=0.6)+
      ggtitle(paste('FA along streamlines', toString(i), sep= ' '))+
      theme_classic()+
      scale_color_manual(values=c('chartreuse1','blueviolet'))+
      scale_fill_manual(values=c('chartreuse1','blueviolet'))+
      theme(legend.position='top')+
      theme(axis.text.x = element_text(face='bold',  size=14, angle=0),
            axis.text.y = element_text(face='bold', size=14, angle=0),
            axis.line.x = element_line(colour = 'black', size=0.5),
            axis.line.y = element_line(colour = 'black', size=0.5),
            # panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            # panel.border = element_blank(),
            panel.background = element_blank())
    ggsave(path_join(list(outputpath,paste(regions_str,'_pred_allbundle', toString(i), '_FA.pdf',sep=''))), plot = last_plot(), device = 'pdf',
           scale = 1, width = 5, height = 5, units = c('in'),dpi = 300)
    
  }
  
}

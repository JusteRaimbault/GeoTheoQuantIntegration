setwd(paste0(Sys.getenv('CS_HOME'),'/QuantitativeEpistemology/GeoTheoQuantIntegration/Models/KDIntegration'))

library(dplyr, warn.conflicts = F)
library(igraph, warn.conflicts = F)
library(glue)
library(reshape2)
library(ggplot2)
library(heatmaply)
library(readr)

source(paste0(Sys.getenv('CS_HOME'),'/Organisation/Models/Utils/R/plots.R'))
source('functions.R')

#1 - Get core for EvUrbTh

edge_file = '../../Data/Corpuses/evurbth_links.csv'
node_file = '../../Data/Corpuses/evurbth.csv'
edges <- read.csv(edge_file,sep=";",header=F,colClasses = c('character','character'))
nodes <- as_tibble(read.csv(node_file,sep=";",stringsAsFactors = F,quote = '"',colClasses = rep('character',4)))
nodes[,3]=as.numeric(unlist(nodes[,3])) # year
nodes[,4]=as.numeric(unlist(nodes[,4])) # depth
citation <- graph_from_data_frame(edges,vertices = nodes)
citation = induced_subgraph(citation,which(components(citation)$membership==1))
# keep at least degree 3 (-> 1252 to annotate)
citationcorehigher = induced_subgraph(citation,which(degree(citation)>2))
while(length(which(degree(citationcorehigher)==1))>0){citationcorehigher = induced_subgraph(citationcorehigher,which(degree(citationcorehigher)>1))}

# export
export_gml(citationcorehigher,'../../Data/Processed/core_evurbth.gml')
write.csv(data.frame(title=V(citationcorehigher)$title,id=V(citationcorehigher)$name,year=V(citationcorehigher)$year),file='../../Data/Corpuses/evurbth_core.csv',row.names = F)


#2 - Get core for Zipf

#2.1 - Test old corpus
nodes <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/res/citation.csv',delim = ";",col_names = c("title","id","year"))
edges <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/res/citation_links.csv',delim = ";",col_names = c('from','to'),col_types = "cc")
basecorpus <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/data/corpus.csv',delim=";",quote="",col_names = c("title","id"),col_types = "cc")
basecorpus = basecorpus[!is.na(basecorpus$id),]
#basecorpus=rbind(basecorpus,c(title="Human behavior and the principle of least effort: An introduction to human ecology",id="14883856614127053528"))
write_delim(basecorpus,file = '../../Data/OriginCorpuses/Zipf.csv',delim = ";",quote = 'all',col_names = F)

edges=edges[sapply(edges$to,FUN = function(id){id%in%basecorpus$id}),] # citations to level 1
#edges[apply(edges,MARGIN = 1,FUN = function(link){link[1]%in%basecorpus$id &&link[2]=="14883856614127053528"}),] # no cit to zipf! no need to add
nodes=nodes[sapply(nodes$id,function(id){id%in%edges$from||id%in%edges$to}),]
# not concluding!
# -> imported basecorpus into bibliodata, construct the citation network the same way as evurbth




#3 - Analysis

  CORPUS='evurbth'

  # load annotated data
  g <- read_graph(file=paste0('../../Data/Processed/core_',CORPUS,'.gml'),format='gml')
  domains <- read_csv(paste0('../../Data/Corpuses/',CORPUS,'_core_KD-ANNOTATED.csv'),col_names = F,col_types = "cccc")
  V(g)$domain = unlist(domains[,4])
  
  #3.1 - visu network -> gephi


  #3.2 - community detection, modularity

  A=as_adjacency_matrix(citationcorehigher,sparse = T)
  M = A+t(A)
  undirected_core = graph_from_adjacency_matrix(M,mode="undirected")
  set.seed(666)
  com = cluster_louvain(undirected_core)
  directedmodularity(com$membership,A)

  d=degree(g,mode='in')
  for(c in unique(com$membership)){
    show(paste0("Community ",c, " ; corpus prop ",100*length(which(com$membership==c))/vcount(undirected_core)))
    currentd=d[com$membership==c];dth=sort(currentd,decreasing = T)[10]
    show(data.frame(titles=V(g)$title[com$membership==c&d>dth],degree=d[com$membership==c&d>dth]))
  }
  citcomnames = list('1'='CA','2'='ABM','3'='UrbanSystems','4'='GIS','5'='LUTI','6'='Complexity','7'='Simpop', '8'='Resilience','9'='GeoStruct','10'='NA')
  comsizes=list()
  for(k in names(citcomnames)){comsizes[k]=length(which(com$membership==as.numeric(k)))}
  comsizes=unlist(comsizes)
  largestcoms = names(sort(comsizes[comsizes>50],decreasing = T))
  
  lp = getLogCitationFlows(g,com,largestcoms,citcomnames)
  
  heatmaply(lp,plot_method = "ggplot")#,file = '../../Results/$CORPUS_communities-citations.png')
  
  # - proportion of KDs in each community
  V(g)$community = unlist(sapply(as.character(com$membership),function(k){if(k%in%largestcoms){citcomnames[k]}else{"Other"}}))
  
  # - diversity within modularity classes
  d = tibble(domain=V(g)$domain,community=V(g)$community) %>% na.omit()
  props = d %>% group_by(community) %>%
    summarise(theory = sum(domain=='theory')/n(),model=sum(domain=='model')/n(),empirical=sum(domain=='empirical')/n(),
              tool = sum(domain=='tool')/n(),data=sum(domain=='data')/n(),method=sum(domain=='method')/n())
  props = rbind(props,c(community='all',theory=sum(d$domain=='theory')/nrow(d),model=sum(d$domain=='model')/nrow(d),empirical=sum(d$domain=='empirical')/nrow(d),tool=sum(d$domain=='tool')/nrow(d),data=sum(d$domain=='data')/nrow(d),method=sum(d$domain=='method')/nrow(d)))
  for(j in 2:ncol(props)){props[,j]=as.numeric(unlist(props[,j]))}
  
  p = melt(props,measure.vars = 2:7)
  p$value=as.numeric(p$value)
  ggsave(
    plot=ggplot(p,aes(x=factor(community,c('all',unique(d$community))),y=value,fill=variable))+geom_col()+ylab("Proportion")+xlab('Community')+stdtheme,
    filename = paste0('../../Results/',CORPUS,'_proportions.png'),width = 30,height=20,units = 'cm'
  )
  
  
  # - modularity of KD classif (compared to shuffled and communities)
  doms = domains$X4
  doms[is.na(doms)]="NA"
  directedmodularity(doms,A) # evurbth = 0.07230344
  # null model
  null_mods = c()
  for(b in 1:100){null_mods=append(null_mods,directedmodularity(sample.int(6,nrow(A),replace = T),A))}
  mean(null_mods)
  sd(null_mods)
  # evurbth: -0.0008 +- 0.0049


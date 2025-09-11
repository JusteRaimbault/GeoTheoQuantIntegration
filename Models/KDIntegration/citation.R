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

#2.2 - Get core
edge_file = '../../Data/Corpuses/zipf_links.csv'
node_file = '../../Data/Corpuses/zipf.csv'
edges <- read.csv(edge_file,sep=";",header=F,colClasses = c('character','character'))
nodes <- as_tibble(read.csv(node_file,sep=";",stringsAsFactors = F,quote = '"',colClasses = rep('character',4)))
nodes[,3]=as.numeric(unlist(nodes[,3])) # year
nodes[,4]=as.numeric(unlist(nodes[,4])) # depth
citation <- graph_from_data_frame(edges,vertices = nodes)
citation = induced_subgraph(citation,which(components(citation)$membership==1))
# need more filtering as everything was exported
basenodes=V(citation)[V(citation)$name%in%basecorpus$id]
keptvertices=basenodes
adjs = adjacent_vertices(citation,basenodes,mode = 'in')
for(adj in names(adjs)){
  keptvertices=c(keptvertices,adjs[[adj]])
  adjs2 = adjacent_vertices(citation,adjs[[adj]],mode = 'in')
  for(adj2 in names(adjs2)){keptvertices=c(keptvertices,adjs2[[adj2]])}
}
keptvertices=unique(keptvertices)
citation=induced_subgraph(citation,keptvertices)
# keep at least degree 30 (included) (-> 858 to annotate)
citationcorehigher = induced_subgraph(citation,which(degree(citation)>29))
while(length(which(degree(citationcorehigher)==1))>0){citationcorehigher = induced_subgraph(citationcorehigher,which(degree(citationcorehigher)>1))}

# export
export_gml(citationcorehigher,'../../Data/Processed/core_zipf.gml')
write.csv(data.frame(title=V(citationcorehigher)$title,id=V(citationcorehigher)$name,year=V(citationcorehigher)$year),file='../../Data/Corpuses/zipf_core.csv',row.names = F)


#3 - Analysis

  #CORPUS='evurbth'
  CORPUS='zipf'

  # load annotated data
  g <- read_graph(file=paste0('../../Data/Processed/core_',CORPUS,'.gml'),format='gml')
  domains <- read_csv(paste0('../../Data/Corpuses/',CORPUS,'_core_KD-ANNOTATED.csv'),col_names = F,col_types = "cccc")
  V(g)$domain = unlist(domains[,4])
  if(CORPUS=='zipf'){# no tool!
    dom_fullnames=c("e"="empirical","th"="theory","mo"="model","me"="method","NA"="NA","d"="data")
    V(g)$domain[is.na(V(g)$domain)]="NA"
    V(g)$domain=dom_fullnames[V(g)$domain]
    domains$domain = V(g)$domain
    V(g)$domain[V(g)$domain=="NA"]=NA
  }
  if(CORPUS=='evurbth'){
    domains$domain = domains$X4
  }
  table(V(g)$domain) # counts
  nrow(domains)-sum(table(V(g)$domain)) # NAs
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
  if(CORPUS=='evurbth'){
    citcomnames = list('1'='CA','2'='ABM','3'='UrbanSystems','4'='GIS','5'='LUTI','6'='Complexity','7'='Simpop', '8'='Resilience','9'='GeoStruct','10'='NA')
  }
  if(CORPUS=='zipf'){
    citcomnames = list('1'='CitySize','2'='Method','3'='Simulation','4'='NEG','5'='Finance','6'='CityScience','7'='NA')
  }
  comsizes=list()
  for(k in names(citcomnames)){comsizes[k]=length(which(com$membership==as.numeric(k)))}
  comsizes=unlist(comsizes)
  if(CORPUS=='evurbth'){
    largestcoms = names(sort(comsizes[comsizes>50],decreasing = T))
  }
  if(CORPUS=='zipf'){
    largestcoms = names(sort(comsizes[comsizes>50],decreasing = T))
  }
   
  lp = getLogCitationFlows(g,com,largestcoms,citcomnames)
  
  heatmaply(lp,plot_method = "ggplot")#,file = '../../Results/$CORPUS_communities-citations.png')
  
  # - proportion of KDs in each community
  V(g)$community = unlist(sapply(as.character(com$membership),function(k){if(k%in%largestcoms){citcomnames[k]}else{"Other"}}))
  
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
  
  # - diversity within modularity classes -> Herfindhal = 1-sum p_i^2
  apply(props[2:7],MARGIN = 1,function(probas){1 - sum(unlist(probas)^2)})
  
  # zipf
  #  "CityScience" "CitySize" "Finance" "Method" "NEG"  "Other" "Simulation" "all" 
  #  0.6820712 0.6147611 0.6844530 0.4963504 0.6185481 0.2448980 0.6900856 0.6678222
  
  # evurbth
  # "ABM"  "CA"        "Complexity"   "GIS"  "LUTI"  "Other"   "Simpop"   "UrbanSystems" "all"
  # 0.6563357 0.6179501 0.6667853 0.6235651 0.5872781 0.4863281 0.7042604 0.6463322 0.7024351
  
  # - modularity of KD classif (compared to shuffled and communities)
  doms = domains$X4
  doms[is.na(doms)]="NA"
  directedmodularity(doms,A) # evurbth = 0.07230344 ; zipf = 0.06645351
  # null model
  null_mods = c()
  for(b in 1:100){null_mods=append(null_mods,directedmodularity(sample.int(6,nrow(A),replace = T),A))}
  mean(null_mods)
  sd(null_mods)
  # evurbth: -0.0008 +- 0.0049
  # zipf : -0.00117891 +- 0.004181261
  
  # - citation flow graphs between domains
  d = ends(g,E(g))
  d=as_tibble(d)
  names(d)=c("from","to")
  d = left_join(d,domains[,c("X2","domain")],by=c("from"="X2"))
  d = left_join(d,domains[,c("X2","domain")],by=c("to"="X2"))
  names(d)<-c("V1","V2","from","to")
  ds = d%>% group_by(from,to) %>% summarise(weight=n())
  
  summary_graph = graph_from_data_frame(ds,directed = T)
  
  coords = layout_as_star(summary_graph)
  set.seed(42)
  coords = coords[sample.int(nrow(coords),size=nrow(coords)),]
  V(summary_graph)$x = coords[,1]
  V(summary_graph)$y = coords[,2]
  #curve_multiple(summary_graph)
  png(paste0('../../Results/',CORPUS,'_domaingraph.png'),width=20,height = 20, units='cm',res=300)
  plot(
    summary_graph,
    edge.width = 10*E(summary_graph)$weight/max(E(summary_graph)$weight),
    edge.curved=seq(-0.8, 0.8, length = ecount(summary_graph)),
    vertex.frame.color = NA,
    margin=0.2
  )
  dev.off()
  
  

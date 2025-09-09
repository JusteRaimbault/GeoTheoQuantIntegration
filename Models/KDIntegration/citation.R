setwd(paste0(Sys.getenv('CS_HOME'),'/QuantitativeEpistemology/GeoTheoQuantIntegration/Models/KDIntegration'))

library(dplyr, warn.conflicts = F)
library(igraph, warn.conflicts = F)
library(glue)
library(reshape2)
library(ggplot2)
library(heatmaply)

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
export_gml(citationcorehigher,'../../Data/core_evurbth.gml')
write.csv(data.frame(title=V(citationcorehigher)$title,id=V(citationcorehigher)$name,year=V(citationcorehigher)$year),file='../../Data/Corpuses/evurbth_core.csv',row.names = F)



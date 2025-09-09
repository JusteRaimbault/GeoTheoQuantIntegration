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
export_gml(citationcorehigher,'../../Data/core_evurbth.gml')
write.csv(data.frame(title=V(citationcorehigher)$title,id=V(citationcorehigher)$name,year=V(citationcorehigher)$year),file='../../Data/Corpuses/evurbth_core.csv',row.names = F)


#2 - Get core for Zipf
nodes <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/res/citation.csv',delim = ";",col_names = c("title","id","year"))
edges <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/res/citation_links.csv',delim = ";",col_names = c('from','to'),col_types = "cc")
basecorpus <- read_delim('https://github.com/JusteRaimbault/MetaZipf/raw/refs/heads/master/Network/Models/CitationNetwork/data/corpus.csv',delim=";",quote="",col_names = c("title","id"),col_types = "cc")
basecorpus = basecorpus[!is.na(basecorpus$id),]
#basecorpus=rbind(basecorpus,c(title="Human behavior and the principle of least effort: An introduction to human ecology",id="14883856614127053528"))

edges=edges[sapply(edges$to,FUN = function(id){id%in%basecorpus$id}),] # citations to level 1
#edges[apply(edges,MARGIN = 1,FUN = function(link){link[1]%in%basecorpus$id &&link[2]=="14883856614127053528"}),] # no cit to zipf! no need to add
nodes=nodes[sapply(nodes$id,function(id){id%in%edges$from||id%in%edges$to}),]


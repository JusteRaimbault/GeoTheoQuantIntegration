
- initial corpus:
   * [on remote, launchdb] ; [fixed quote handling in csv import in BiblioData: 0521080e44c] 
   * import initial corpus:
        `java -jar bibliodata.jar --database --import ../../Data/OriginCorpuses/EvUrbTh.csv geotheoquantintegration 2 evurbth`
   * [to check] ! bug, initial depth not taken into account? fine with csv import [add todo]

- collect citation network
   * torpool java -jar torpool.jar 50 9050 --mongo
   * Collect citations [for evolth only]: java -jar bibliodata.jar --citation --mongo geotheoquantintegration 10000
     [ok in <12h]
   * Compute priorities: java -jar bibliodata.jar --database --priority geotheoquantintegration 2 [NOT USEFUL]
   * export : java -jar bibliodata.jar --database --export geotheoquantintegration $EXPORTFILE [fixed export function: why was not working anymore since last Y? (urbsimdigtwins quantep)]

- Zipf corpus -> get 2016 corpus from https://github.com/JusteRaimbault/MetaZipf/tree/master/Network at 1a10af3269b24
TODO reorg repo into dedicated one
! corr parsing erros in csv:
   1  2289     3 a double  "5184614407784841768;1981"  ""   
 2  2289     4 3 columns "4 columns"                 ""   
 3  5947     3 a double  "14482711318377341270;2003" ""   
 4  5947     4 3 columns "4 columns"                 ""   
 5 11139     3 a double  "5945408048924492775;\""    ""   
 6 11139     4 3 columns "4 columns"                 ""   
 7 28805     3 a double  "13768771170206739465;1986" ""   
 8 28805     4 3 columns "4 columns"                 ""   
 9 28830     3 a double  "9651949334057386734;2006"  ""   
10 28830     4 3 columns "4 columns"                 ""  
 -> corpus pas coherent (6 remaining qd filtre sur base_corpus) -> redo collection

 * java -jar bibliodata.jar --database --import ../../Data/OriginCorpuses/Zipf.csv geotheoquantintegration 2 zipf
 * java -jar bibliodata.jar --citation --mongo geotheoquantintegration 10000
 * Stop after 12h~ -> remaining 2015 [on 2025/09/10]
 * Export java -jar bibliodata.jar --database --export geotheoquantintegration ../../Data/Corpuses/zipf
 * [2025/09/16] rename prelim corpus with date, relaunch data collection



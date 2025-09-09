
- initial corpus:
   * [on remote, launchdb] ; [fixed quote handling in csv import in BiblioData: 0521080e44c] 
   * import initial corpus:
        `java -jar bibliodata.jar --database --import ../../Data/OriginCorpuses/EvUrbTh.csv geotheoquantintegration 2 evurbth`
   * [to check] ! bug, initial depth not taken into account? fine with csv import [add todo]

- collect citation network
   * torpool java -jar torpool.jar 50 9050 --mongo
   * Collect citations: java -jar bibliodata.jar --citation --mongo geotheoquantintegration 10000
   * Compute priorities: java -jar bibliodata.jar --database --priority geotheoquantintegration 2 [NOT USEFUL]
   * export : java -jar bibliodata.jar --database --export geotheoquantintegration $EXPORTFILE [fixed export function: why was not working anymore since last Y? (urbsimdigtwins quantep)]




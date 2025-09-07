
- initial corpus:
   * 
   * import initial corpus java -jar bibliodata.jar --keywords --mongo query.csv urbsimdigtwins 100 2 true

   * [to check] ! bug, initial depth not taken into account -> increase depth: twice: java -jar bibliodata.jar --database --incrdepth geotheoquantintegration

- collect citation network
   * torpool java -jar torpool.jar 50 9050 --mongo

   * Collect citations: java -jar bibliodata.jar --citation --mongo geotheoquantintegration 10000
     
   * export : java -jar bibliodata.jar --database --export geotheoquantintegration $EXPORTFILE -1 -1 2 false "" -1




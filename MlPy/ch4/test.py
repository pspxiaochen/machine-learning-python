import feedparser
import bayes
reload(bayes)
ny=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
sf=feedparser.parse('http://sports.yahoo.com/nba/teams/hou/rss.xml')
vocabList,PSF,pNY=bayes.localWords(ny,sf)





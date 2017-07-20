#-*- coding: UTF-8 -*-
import trees
import treePlotter
fr=open('lenses.txt')
lenses=[inst.strip().split('\t') for inst in fr.readlines()]
print lenses
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree=trees.createTree(lenses,lensesLabels)
treePlotter.createPlot(lensesTree)


# 1. run logit and export top N words to csv (DONE)

env/bin/python main.py bag-of-words --strip-named-entities "data/readers/readers.csv" | tee output/logit-bow-readers.log &
env/bin/python main.py bag-of-words --strip-named-entities "data/literature/literature.csv" | tee output/logit-bow-literature.log &
wait
echo "Finished generating CSV files!"
echo

# 2. generate wordcloud images by running bow_overlap_2 (DONE)

echo "Generating word cloud images..."
cd graded_readers_stats/bow_overlap || exit
# set PYTHONPATH to find our `graded_readers_stats.*` modules
PYTHONPATH=../../ ../../env/bin/python bow_overlap_2.py
echo "Finished generating word cloud images..."
echo

# 3. run magick command to assemble the images (DONE)

echo "Combining word cloud images..."
magick convert \
-page 2800x2800+0+0 bg_neg.jpg \
-page +0+0 output/neg/bi1.png \
-page +0+0 output/neg/bi2.png \
-page +0+0 output/neg/bi3.png \
-page +0+0 output/neg/bo1.png \
-page +0+0 output/neg/bo2.png \
-page +0+0 output/neg/bo3.png \
-page +0+0 output/neg/ti1.png \
-page +0+0 output/neg/ti2.png \
-page +0+0 output/neg/ti3.png \
-page +0+0 output/neg/to1.png \
-page +0+0 output/neg/to2.png \
-page +0+0 output/neg/to3.png \
-background white \
-flatten output/neg/output.jpg

magick \
legend_neg.jpg output/neg/output.jpg -geometry +150+420 -composite \
labels.png -composite output/neg/final_neg.jpg

magick convert \
-page 2800x2800+0+0 bg_pos.jpg \
-page +0+0 output/pos/bi1.png \
-page +0+0 output/pos/bi2.png \
-page +0+0 output/pos/bi3.png \
-page +0+0 output/pos/bo1.png \
-page +0+0 output/pos/bo2.png \
-page +0+0 output/pos/bo3.png \
-page +0+0 output/pos/ti1.png \
-page +0+0 output/pos/ti2.png \
-page +0+0 output/pos/ti3.png \
-page +0+0 output/pos/to1.png \
-page +0+0 output/pos/to2.png \
-page +0+0 output/pos/to3.png \
-background white \
-flatten output/pos/output.jpg

magick \
legend_pos.jpg output/pos/output.jpg -geometry +150+420 -composite \
labels.png -composite output/pos/final_pos.jpg

echo "Finished combining word cloud images!"

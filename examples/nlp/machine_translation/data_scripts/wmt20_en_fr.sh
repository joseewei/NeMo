wmt_dir=$1;
out_dir=$2;
script_dir=$(pwd)
bicleaner_model_path=$3;
bifixer_path=$4;
laser_path=$5;
langid_model_path=$6;
moses_path=$7;

mkdir -p ${out_dir}
mkdir -p ${wmt_dir}
mkdir -p ${wmt_dir}/orig

URLS=(
    "http://www.statmt.org/europarl/v10/training/europarl-v10.fr-en.tsv.gz"
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-fr.txt.gz"
    "http://statmt.org/wmt10/training-giga-fren.tar"
    "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-fr.tsv.gz"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EESC2017.en-fr.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2016.en-fr.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/ecb2017.en-fr.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EMA2016.en-fr.tmx.zip"
    "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-fr.tsv.gz"
)

FILES=(
    "europarl-v10.fr-en.tsv.gz"
    "training-parallel-commoncrawl.tgz"
    "en-fr.txt.gz"
    "training-giga-fren.tar"
    "news-commentary-v15.en-fr.tsv.gz"
    "EESC2017.en-fr.tmx.zip"
    "rapid2016.en-fr.tmx.zip"
    "ecb2017.en-fr.tmx.zip"
    "EMA2016.en-fr.tmx.zip"
    "WikiMatrix.en-fr.tsv.gz"
)

URLS_mono_fr=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.fr.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.fr.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.fr.shuffled.v2.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2015.fr.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.fr.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2017.fr.shuffled.gz"
)

URLS_mono_en=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.en.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.en.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.en.shuffled.v2.gz"
    "http://data.statmt.org/wmt16/translation-task/news.2015.en.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.en.shuffled.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.gz"
)

FILES_de=(
    "news.2007.fr.shuffled.gz"
    "news.2008.fr.shuffled.gz"
    "news.2009.fr.shuffled.gz"
    "news.2010.fr.shuffled.gz"
    "news.2011.fr.shuffled.gz"
    "news.2012.fr.shuffled.gz"
    "news.2013.fr.shuffled.gz"
    "news.2014.fr.shuffled.v2.gz"
    "news.2015.fr.shuffled.gz"
    "news.2016.fr.shuffled.gz"
    "news.2017.fr.shuffled.gz"
)

FILES_en=(
    "news.2007.en.shuffled.gz"
    "news.2008.en.shuffled.gz"
    "news.2009.en.shuffled.gz"
    "news.2010.en.shuffled.gz"
    "news.2011.en.shuffled.gz"
    "news.2012.en.shuffled.gz"
    "news.2013.en.shuffled.gz"
    "news.2014.en.shuffled.v2.gz"
    "news.2015.en.shuffled.gz"
    "news.2016.en.shuffled.gz"
    "news.2017.en.shuffled.deduped.gz"
)

OUTDIR=$out_dir
lang1=en
lang2=fr
lang=en-fr
rev_lang=fr-en
orig=${wmt_dir}/orig

mkdir -p $OUTDIR
mkdir -p $OUTDIR/parallel
mkdir -p $OUTDIR/mono

cd $orig

echo "=================================================="
echo "========= Downloading and Unpacking Data ========="
echo "=================================================="

echo "Downloading ...."
for ((i=0;i<${#URLS[@]};++i)); do
    file=${FILES[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS[i]}
        wget "$url"
        if [ -f $file ]; then
            echo "$url successfully downloaded."
        else
            echo "$url not successfully downloaded."
            exit 1
        fi
        if [ ${file: -4} == ".tgz" ]; then
            tar zxvf $file
        elif [ ${file: -4} == ".tar" ]; then
            tar xvf $file
        elif [ ${file: -3} == ".gz" ]; then
            gunzip -k $file
        elif [ ${file: -4} == ".zip" ]; then
            unzip $file
        fi
    fi
done

cd ..

echo "Unpacking ..."
rm $OUTDIR/parallel/*

echo "Adding Europarl v10"
awk -F "\t" '{print $1}' $orig/europarl-v10.fr-en.tsv >> $OUTDIR/parallel/train.$lang.fr
awk -F "\t" '{print $2}' $orig/europarl-v10.fr-en.tsv >> $OUTDIR/parallel/train.$lang.en

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Commoncrawl"
cat $orig/commoncrawl.fr-en.fr >> $OUTDIR/parallel/train.$lang.fr
cat $orig/commoncrawl.fr-en.en >> $OUTDIR/parallel/train.$lang.en

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding News Commentary v15"
awk -F "\t" '{print $1}' $orig/news-commentary-v15.en-fr.tsv >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/news-commentary-v15.en-fr.tsv >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Paracrawl 7.1"
awk -F "\t" '{print $1}' $orig/en-fr.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/en-fr.txt >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Giga French"
gunzip -k -f $orig/giga-fren.release2.fixed.en.gz
gunzip -k -f $orig/giga-fren.release2.fixed.fr.gz

cat $orig/giga-fren.release2.fixed.en >> $OUTDIR/parallel/train.$lang.en
cat $orig/giga-fren.release2.fixed.fr >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding WikiMatrix v1"
python $laser_path/extract.py \
    --tsv $orig/WikiMatrix.en-fr.tsv.gz \
    --bitext $orig/WikiMatrix.en-fr.txt \
    --src-lang en \
    --trg-lang fr \
    --threshold 1.04

fasttext predict $langid_model_path $orig/WikiMatrix.en-fr.txt.en > $orig/WikiMatrix.en-fr.txt.en.langid
fasttext predict $langid_model_path $orig/WikiMatrix.en-fr.txt.fr > $orig/WikiMatrix.en-fr.txt.fr.langid

paste -d "\t" $orig/WikiMatrix.en-fr.txt.en $orig/WikiMatrix.en-fr.txt.fr $orig/WikiMatrix.en-fr.txt.en.langid $orig/WikiMatrix.en-fr.txt.fr.langid | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__fr") {print $1}}' >> $OUTDIR/parallel/train.$lang.en
paste -d "\t" $orig/WikiMatrix.en-fr.txt.en $orig/WikiMatrix.en-fr.txt.fr $orig/WikiMatrix.en-fr.txt.en.langid $orig/WikiMatrix.en-fr.txt.fr.langid | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__fr") {print $2}}' >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding EESC2017"
python $script_dir/tmx2txt.py --codelist en,fr $orig/EESC.en-fr.tmx $orig/EESC.en-fr.tmx.txt
awk -F "\t" '{print $1}' $orig/EESC.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EESC.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Rapid2019"
python $script_dir/tmx2txt.py --codelist en,fr $orig/rapid2016.en-fr.tmx $orig/rapid2016.en-fr.tmx.txt
awk -F "\t" '{print $1}' $orig/rapid2016.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/rapid2016.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding ECB2017"
python $script_dir/tmx2txt.py --codelist en,fr $orig/ecb2017.UNIQUE.en-fr.tmx $orig/ecb2017.UNIQUE.en-fr.tmx.txt
awk -F "\t" '{print $1}' $orig/ecb2017.UNIQUE.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/ecb2017.UNIQUE.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding EMEA2016"
python $script_dir/tmx2txt.py --codelist en,fr $orig/EMEA2016.en-fr.tmx $orig/EMEA2016.en-fr.tmx.txt
awk -F "\t" '{print $1}' $orig/EMEA2016.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EMEA2016.en-fr.tmx.txt >> $OUTDIR/parallel/train.$lang.fr

wc -l $OUTDIR/parallel/train.$lang.fr
wc -l $OUTDIR/parallel/train.$lang.en

echo "Fetching Validation data $lang" 
sacrebleu -t wmt13 -l $lang --echo src > ${OUTDIR}/parallel/newstest2013-$lang.src
sacrebleu -t wmt13 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2013-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt14 -l $lang --echo src > ${OUTDIR}/parallel/newstest2014-$lang.src
sacrebleu -t wmt14 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2014-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt13 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2013-$rev_lang.src
sacrebleu -t wmt13 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2013-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt14 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2014-$rev_lang.src
sacrebleu -t wmt14 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2014-$rev_lang.ref

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

echo "Filtering data based on max length and length ratio ..."
$moses_path/scripts/training/clean-corpus-n.perl -ratio 1.3 ${OUTDIR}/parallel/train.$lang $lang1 $lang2 ${OUTDIR}/parallel/train.$lang.filter 1 250

echo "Applying bi-cleaner classifier"
awk '{print "-\t-"}' $OUTDIR/parallel/train.$lang.filter.en \
| paste -d "\t" - $OUTDIR/parallel/train.$lang.filter.en $OUTDIR/parallel/train.$lang.filter.fr \
| bicleaner-classify - - $bicleaner_model_path > $OUTDIR/parallel/train.$lang.bicleaner.score

echo "Applying bifixer & dedup"
cat $OUTDIR/parallel/train.$lang.bicleaner.score \
| parallel -j 19 --pipe -k -l 30000 python $bifixer_path/bifixer.py \
    --ignore_segmentation -q - - en fr \
    | awk -F "\t" '!seen[$6]++' - > $OUTDIR/parallel/train.$lang.bifixer.score

echo "Creating data with different classifier confidence values ..."
awk -F "\t" '{print $3}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.en
awk -F "\t" '{print $4}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.fr

awk -F "\t" '{ if ($5>0.5) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.en
awk -F "\t" '{ if ($5>0.5) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.fr

awk -F "\t" '{ if ($5>0.6) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.en
awk -F "\t" '{ if ($5>0.6) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.fr

awk -F "\t" '{ if ($5>0.75) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.en
awk -F "\t" '{ if ($5>0.75) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.fr

echo "=================================================="
echo "========== Moses Punct Normalization ============="
echo "=================================================="

echo "Normalizing punct ..."
for t in all 50 60 75; do
    for l in $lang1 $lang2; do
        cat $OUTDIR/parallel/train.$lang.$t.$l | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $l | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > $OUTDIR/parallel/train.clean.$lang.$t.$l
        cat $OUTDIR/parallel/train.clean.$lang.$t.$l | perl $moses_path/scripts/tokenizer/tokenizer.perl -l $l -no-escape -threads 20 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$l
    done
    cat $OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.common
    cat $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.common

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.$lang2.shuffled

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2.shuffled
done

echo "Normalizing valid/test punct ..."

for t in newstest2013 newstest2014; do
    cat ${OUTDIR}/parallel/$t-$lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.src
    cat ${OUTDIR}/parallel/$t-$lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l fr | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l fr > ${OUTDIR}/parallel/$t-$lang.clean.tok.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l fr | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l fr > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.ref
done

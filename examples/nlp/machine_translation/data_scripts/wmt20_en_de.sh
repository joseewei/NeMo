wmt_dir=$1;
out_dir=$2;
script_dir=$(pwd)

mkdir -p ${out_dir}
mkdir -p ${wmt_dir}
mkdir -p ${wmt_dir}/orig

URLS=(
    "http://www.statmt.org/europarl/v10/training/europarl-v10.de-en.tsv.gz"
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-de.txt.gz"
    "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.de-en.tsv.gz"
    "http://data.statmt.org/wikititles/v2/wikititles-v2.de-en.tsv.gz"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EESC2017.de-en.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2019.de-en.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/ecb2017.de-en.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EMA2016.de-en.tmx.zip"
    "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.de-en.langid.tsv.gz"
)

FILES=(
    "europarl-v10.de-en.tsv"
    "training-parallel-commoncrawl.tgz"
    "en-de.txt.gz"
    "news-commentary-v15.de-en.tsv.gz"
    "wikititles-v2.de-en.tsv.gz"
    "EESC2017.de-en.tmx.zip"
    "rapid2019.de-en.tmx.zip"
    "ecb2017.de-en.tmx.zip"
    "EMA2016.de-en.tmx.zip"
    "WikiMatrix.v1.de-en.langid.tsv.gz"
)

CORPORA=(
    "training/europarl-v7.de-en"
    "commoncrawl.de-en"
    "training-parallel-nc-v13/news-commentary-v13.de-en"
    "rapid2016.de-en"
)

URLS_mono_de=(
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2007.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2008.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2009.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2010.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2011.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2012.de.shuffled.gz"
    "http://www.statmt.org/wmt14/training-monolingual-news-crawl/news.2013.de.shuffled.gz"
    "http://www.statmt.org/wmt15/training-monolingual-news-crawl-v2/news.2014.de.shuffled.v2.gz"
    "http://data.statmt.org/wmt16/translation-task/news.2015.de.shuffled.gz"
    "http://data.statmt.org/wmt17/translation-task/news.2016.de.shuffled.gz"
    "http://data.statmt.org/wmt18/translation-task/news.2017.de.shuffled.deduped.gz"
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
    "http://data.statmt.org/wmt18/translation-task/news.2017.en.shuffled.deduped.gz"
)

FILES_de=(
    "news.2007.de.shuffled.gz"
    "news.2008.de.shuffled.gz"
    "news.2009.de.shuffled.gz"
    "news.2010.de.shuffled.gz"
    "news.2011.de.shuffled.gz"
    "news.2012.de.shuffled.gz"
    "news.2013.de.shuffled.gz"
    "news.2014.de.shuffled.v2.gz"
    "news.2015.de.shuffled.gz"
    "news.2016.de.shuffled.gz"
    "news.2017.de.shuffled.deduped.gz"
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
lang2=de
lang=en-de
rev_lang=de-en
orig=${wmt_dir}/orig

mkdir -p $OUTDIR
mkdir -p $OUTDIR/parallel
mkdir -p $OUTDIR/mono

cd $orig

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

echo "pre-processing train data..."
rm $OUTDIR/parallel/*

echo "Adding Europarl v10"
awk -F "\t" '{print $1}' $orig/europarl-v10.de-en.tsv >> $OUTDIR/parallel/train.$lang.de
awk -F "\t" '{print $2}' $orig/europarl-v10.de-en.tsv >> $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding Commoncrawl"
cat $orig/commoncrawl.de-en.de >> $OUTDIR/parallel/train.$lang.de
cat $orig/commoncrawl.de-en.en >> $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding News Commentary v15"
awk -F "\t" '{print $1}' $orig/news-commentary-v15.de-en.tsv >> $OUTDIR/parallel/train.$lang.de
awk -F "\t" '{print $2}' $orig/news-commentary-v15.de-en.tsv >> $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding Paracrawl 7.1"
awk -F "\t" '{print $1}' $orig/en-de.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/en-de.txt >> $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding Wikititles v2"
awk -F "\t" '{print $1}' $orig/wikititles-v2.de-en.tsv >> $OUTDIR/parallel/train.$lang.de
awk -F "\t" '{print $2}' $orig/wikititles-v2.de-en.tsv >> $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de


echo "Adding WikiMatrix v1"
awk -F "\t" '{ if ($4 == "de" && $5 == "en") {print $2}}' $orig/WikiMatrix.v1.de-en.langid.tsv >> $OUTDIR/parallel/train.$lang.de
awk -F "\t" '{ if ($4 == "de" && $5 == "en") {print $3}}' $orig/WikiMatrix.v1.de-en.langid.tsv >> $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding EESC2017"
python $script_dir/tmx2txt.py --codelist en,de $orig/EESC.de-en.tmx $orig/EESC.de-en.tmx.txt
awk -F "\t" '{print $1}' $orig/EESC.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EESC.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding Rapid2019"
python $script_dir/tmx2txt.py --codelist en,de $orig/RAPID_2019.UNIQUE.de-en.tmx $orig/RAPID_2019.UNIQUE.de-en.tmx.txt
awk -F "\t" '{print $1}' $orig/RAPID_2019.UNIQUE.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/RAPID_2019.UNIQUE.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding ECB2017"
python $script_dir/tmx2txt.py --codelist en,de $orig/ecb2017.UNIQUE.de-en.tmx $orig/ecb2017.UNIQUE.de-en.tmx.txt
awk -F "\t" '{print $1}' $orig/ecb2017.UNIQUE.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/ecb2017.UNIQUE.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

echo "Adding EMA2016"
python $script_dir/tmx2txt.py --codelist en,de $orig/EMEA2016.de-en.tmx $orig/EMEA2016.de-en.tmx.txt
awk -F "\t" '{print $1}' $orig/EMEA2016.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EMEA2016.de-en.tmx.txt >> $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de
wc -l $OUTDIR/parallel/train.$lang.de

if [ ! -f clean-corpus-n.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
    chmod +x clean-corpus-n.perl
fi

if [ ! -f normalize-punctuation.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/normalize-punctuation.perl
    chmod +x normalize-punctuation.perl
fi

if [ ! -f remove-non-printing-char.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/tokenizer/remove-non-printing-char.perl
    chmod +x remove-non-printing-char.perl
fi

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

echo "Cleaning data ..."
for l in $lang1 $lang2; do
    cat $OUTDIR/parallel/train.$lang.$l | perl normalize-punctuation.perl -l $l | perl remove-non-printing-char.perl > $OUTDIR/parallel/train.clean.$lang.$l
done

cat ${OUTDIR}/parallel/newstest2013-$lang.src | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$lang.clean.src
cat ${OUTDIR}/parallel/newstest2013-$lang.ref | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$lang.clean.ref

cat ${OUTDIR}/parallel/newstest2014-$lang.src | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$lang.clean.src
cat ${OUTDIR}/parallel/newstest2014-$lang.ref | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$lang.clean.ref

cat ${OUTDIR}/parallel/newstest2013-$rev_lang.src | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$rev_lang.clean.src
cat ${OUTDIR}/parallel/newstest2013-$rev_lang.ref | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2013-$rev_lang.clean.ref

cat ${OUTDIR}/parallel/newstest2014-$rev_lang.src | perl normalize-punctuation.perl -l de | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$rev_lang.clean.src
cat ${OUTDIR}/parallel/newstest2014-$rev_lang.ref | perl normalize-punctuation.perl -l en | perl remove-non-printing-char.perl > ${OUTDIR}/parallel/newstest2014-$rev_lang.clean.ref

echo "Filtering data ..."
./clean-corpus-n.perl -ratio 1.3 ${OUTDIR}/parallel/train.clean.$lang $lang1 $lang2 ${OUTDIR}/parallel/train.clean.filter 1 250

echo 'Shuffling'
shuf --random-source=${OUTDIR}/parallel/train.clean.filter.$lang1 ${OUTDIR}/parallel/train.clean.filter.$lang1 > ${OUTDIR}/parallel/train.clean.filter.$lang1.shuffled
shuf --random-source=${OUTDIR}/parallel/train.clean.filter.$lang2 ${OUTDIR}/parallel/train.clean.filter.$lang2 > ${OUTDIR}/parallel/train.clean.filter.$lang2.shuffled
cat ${OUTDIR}/parallel/train.clean.filter.$lang1.shuffled ${OUTDIR}/parallel/train.filter.clean.$lang2.shuffled > ${OUTDIR}/parallel/train.clean.filter.$lang.shuffled.common

OUTDIR_MONO=$OUTDIR/mono/
mkdir -p $OUTDIR_MONO

cd $orig

echo "Done Processing Parallel Corpus, Fetching Monolingual Data ..."

echo "Fetching English Monolingual data ..."

for ((i=0;i<${#URLS_mono_en[@]};++i)); do
    file=${FILES_en[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS_mono_en[i]}
        wget "$url"
    fi
done

if [ -f ${OUTDIR_MONO}/monolingual.news.en ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES_en[@]}"; do echo $orig/$FILE; done) > $OUTDIR_MONO/monolingual.news.en
fi

echo "Deduplicating data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.en ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    awk '!a[$0]++' ${OUTDIR_MONO}/monolingual.news.en > ${OUTDIR_MONO}/monolingual.news.dedup.en
fi

echo "Fetching German Monolingual data ..."

cd $orig

for ((i=0;i<${#URLS_mono_de[@]};++i)); do
    file=${FILES_de[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS_mono_de[i]}
        wget "$url"
    fi
done

echo "Subsampling data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.de ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES_de[@]}"; do echo $orig/$FILE; done) > ${OUTDIR_MONO}/monolingual.news.de
fi

echo "Deduplicating data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.de ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    awk '!a[$0]++' ${OUTDIR_MONO}/monolingual.news.de > ${OUTDIR_MONO}/monolingual.news.dedup.de
fi

echo "Cleaning data ..."
for l in $lang1 $lang2; do
    cat ${OUTDIR_MONO}/monolingual.news.dedup.$l | perl normalize-punctuation.perl -l $l | perl remove-non-printing-char.perl > ${OUTDIR_MONO}/monolingual.news.dedup.clean.$l
done

wmt_dir=$1;
out_dir=$2;

mkdir -p ${out_dir}
mkdir -p ${wmt_dir}
mkdir -p ${wmt_dir}/orig

URLS=(
    "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-zh.tsv.gz"
    "http://data.statmt.org/wikititles/v2/wikititles-v2.zh-en.tsv.gz"
    "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-zh.langid.tsv.gz"
    "http://data.statmt.org/wmt20/translation-task/back-translation/zh-en/news.translatedto.zh.gz"
    "http://data.statmt.org/wmt20/translation-task/back-translation/zh-en/news.en.gz"
)

FILES=(
    "news-commentary-v15.en-zh.tsv.gz"
    "wikititles-v2.zh-en.tsv.gz"
    "WikiMatrix.v1.en-zh.langid.tsv.gz"
    "news.translatedto.zh.gz"
    "news.en.gz"
)

CORPORA=(
    "news-commentary-v15"
    "wikititles-v2"
    "WikiMatrix.v1"
    "backtranslated-news-zh"
    "backtranslated-news-en"
)

URLS_mono_zh=(
    "http://data.statmt.org/news-crawl/zh/news.2008.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2010.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2011.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2012.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2013.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2014.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2015.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2016.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2017.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2018.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/zh/news.2019.zh.shuffled.deduped.gz"
    "http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.zh.gz"
)

URLS_mono_en=(
    "http://data.statmt.org/news-crawl/en/news.2007.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2008.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2009.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2010.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2011.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2012.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2013.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2014.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2015.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2016.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2017.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2018.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-crawl/en/news.2019.en.shuffled.deduped.gz"
    "http://data.statmt.org/news-commentary/v15/training-monolingual/news-commentary-v15.en.gz"
)

FILES_zh=(
    "news.2007.zh.shuffled.deduped.gz"
    "news.2010.zh.shuffled.deduped.gz"
    "news.2011.zh.shuffled.deduped.gz"
    "news.2012.zh.shuffled.deduped.gz"
    "news.2013.zh.shuffled.deduped.gz"
    "news.2014.zh.shuffled.deduped.gz"
    "news.2015.zh.shuffled.deduped.gz"
    "news.2016.zh.shuffled.deduped.gz"
    "news.2017.zh.shuffled.deduped.gz"
    "news.2018.zh.shuffled.deduped.gz"
    "news.2019.zh.shuffled.deduped.gz"
    "news-commentary-v15.zh.gz"
)

FILES_en=(
    "news.2007.en.shuffled.deduped.gz"
    "news.2008.en.shuffled.deduped.gz"
    "news.2009.en.shuffled.deduped.gz"
    "news.2010.en.shuffled.deduped.gz"
    "news.2011.en.shuffled.deduped.gz"
    "news.2012.en.shuffled.deduped.gz"
    "news.2013.en.shuffled.deduped.gz"
    "news.2014.en.shuffled.deduped.gz"
    "news.2015.en.shuffled.deduped.gz"
    "news.2016.en.shuffled.deduped.gz"
    "news.2017.en.shuffled.deduped.gz"
    "news.2018.en.shuffled.deduped.gz"
    "news.2019.en.shuffled.deduped.gz"
    "news-commentary-v15.en.gz"
)

OUTDIR=$out_dir
lang1=en
lang2=zh
lang=en-zh
rev_lang=zh-en
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
        if [ ${file: -3} == ".gz" ]; then
            gunzip -k $file
        fi
    fi
done

cd ..

echo "pre-processing train data..."
rm $OUTDIR/parallel/*

if [ -f $orig/UNv1.0.6way.tar.gz.00 ]
then
    echo "Adding UN Parallel Corpus"
    cat $orig/UNv1.0.6way.tar.gz.* > $orig/UNv1.0.6way.tar.gz
    tar -xzf $orig/UNv1.0.6way.tar.gz
    for l in $lang1 $lang2; do
        cat $orig/6way/UNv1.0.6way.$l >> $OUTDIR/parallel/train.$lang.$l
    done
fi

if [ -f $orig/news-commentary-v15.en-zh.tsv ]
then
    echo "Adding news commentary"
    awk -F "\t" '{print $1}' $orig/news-commentary-v15.en-zh.tsv >> $OUTDIR/parallel/train.$lang.en
    awk -F "\t" '{print $2}' $orig/news-commentary-v15.en-zh.tsv >> $OUTDIR/parallel/train.$lang.zh
fi

if [ -f $orig/news.translatedto.zh ]
then
    echo "Adding WMT provided backtranslated data"
    cat $orig/news.translatedto.zh >> $OUTDIR/parallel/train.$lang.zh
    cat $orig/news.en >> $OUTDIR/parallel/train.$lang.en
fi

if [ -f $orig/wikititles-v2.zh-en.tsv ]
then
    echo "Adding Wiki titles"
    awk -F "\t" '{print $1}' $orig/wikititles-v2.zh-en.tsv >> $OUTDIR/parallel/train.$lang.zh
    awk -F "\t" '{print $2}' $orig/wikititles-v2.zh-en.tsv >> $OUTDIR/parallel/train.$lang.en
fi

if [ -f $orig/WikiMatrix.v1.en-zh.langid.tsv ]
then
    echo "Adding Wiki Matrix"
    awk -F "\t" '{ if ($4 == "en" && $5 == "zh") {print $2}}' $orig/WikiMatrix.v1.en-zh.langid.tsv >> $OUTDIR/parallel/train.$lang.en
    awk -F "\t" '{ if ($4 == "en" && $5 == "zh") {print $2}}' $orig/WikiMatrix.v1.en-zh.langid.tsv >> $OUTDIR/parallel/train.$lang.zh
fi

if [ ! -f clean-corpus-n.perl ]
then
    wget https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/training/clean-corpus-n.perl
    chmod +x clean-corpus-n.perl
fi

./clean-corpus-n.perl -ratio 1000 ${OUTDIR}/parallel/train.$lang $lang1 $lang2 ${OUTDIR}/parallel/train.clean 1 250

echo 'Shuffling'
shuf --random-source=${OUTDIR}/parallel/train.clean.$lang1 ${OUTDIR}/parallel/train.clean.$lang1 > ${OUTDIR}/parallel/train.clean.$lang1.shuffled
shuf --random-source=${OUTDIR}/parallel/train.clean.$lang1 ${OUTDIR}/parallel/train.clean.$lang2 > ${OUTDIR}/parallel/train.clean.$lang2.shuffled
cat ${OUTDIR}/parallel/train.clean.$lang1.shuffled ${OUTDIR}/parallel/train.clean.$lang2.shuffled > ${OUTDIR}/parallel/train.clean.$lang.shuffled.common

echo "Fetching Validation data $lang" 
sacrebleu -t wmt19 -l $lang --echo src > ${OUTDIR}/parallel/wmt19-$lang.src
sacrebleu -t wmt19 -l $lang --echo ref > ${OUTDIR}/parallel/wmt19-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt20 -l $lang --echo src > ${OUTDIR}/parallel/wmt20-$lang.src
sacrebleu -t wmt20 -l $lang --echo ref > ${OUTDIR}/parallel/wmt20-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt19 -l $rev_lang --echo src > ${OUTDIR}/parallel/wmt19-$rev_lang.src
sacrebleu -t wmt19 -l $rev_lang --echo ref > ${OUTDIR}/parallel/wmt19-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt20 -l $rev_lang --echo src > ${OUTDIR}/parallel/wmt20-$rev_lang.src
sacrebleu -t wmt20 -l $rev_lang --echo ref > ${OUTDIR}/parallel/wmt20-$rev_lang.ref

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

echo "Fetching Chinese Monolingual data ..."

cd $orig

for ((i=0;i<${#URLS_mono_zh[@]};++i)); do
    file=${FILES_zh[i]}
    if [ -f $file ]; then
        echo "$file already exists, skipping download"
    else
        url=${URLS_mono_zh[i]}
        wget "$url"
    fi
done

echo "Unzipping data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.zh ]; then
    echo "found monolingual sample, skipping shuffle/sample/tokenize"
else
    gzip -c -d -k $(for FILE in "${FILES_zh[@]}"; do echo $orig/$FILE; done) > ${OUTDIR_MONO}/monolingual.news.zh
fi

echo "Deduplicating data ..."
if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.zh ]; then
    echo "found deduplicated monolingual sample, skipping deduplication step"
else
    awk '!a[$0]++' ${OUTDIR_MONO}/monolingual.news.zh > ${OUTDIR_MONO}/monolingual.news.dedup.zh
fi

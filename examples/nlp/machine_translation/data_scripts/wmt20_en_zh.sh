wmt_dir=$1;
out_dir=$2;
script_dir=$(pwd)
langid_model_path=$3;
moses_path=$4;

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
    "news.2008.zh.shuffled.deduped.gz"
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

echo "=================================================="
echo "========= Downloading and Unpacking Data ========="
echo "=================================================="

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

wc -l $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.zh

if [ -f $orig/news-commentary-v15.en-zh.tsv ]
then
    echo "Adding news commentary"
    awk -F "\t" '{print $1}' $orig/news-commentary-v15.en-zh.tsv >> $OUTDIR/parallel/train.$lang.en
    awk -F "\t" '{print $2}' $orig/news-commentary-v15.en-zh.tsv >> $OUTDIR/parallel/train.$lang.zh
fi

wc -l $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.zh

if [ -f $orig/news.translatedto.zh ]
then
    echo "Adding WMT provided backtranslated data"
    cat $orig/news.translatedto.zh >> $OUTDIR/parallel/train.$lang.zh
    cat $orig/news.en >> $OUTDIR/parallel/train.$lang.en
fi

wc -l $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.zh

if [ -f $orig/wikititles-v2.zh-en.tsv ]
then
    echo "Adding Wiki titles"
    awk -F "\t" '{print $1}' $orig/wikititles-v2.zh-en.tsv >> $OUTDIR/parallel/train.$lang.zh
    awk -F "\t" '{print $2}' $orig/wikititles-v2.zh-en.tsv >> $OUTDIR/parallel/train.$lang.en
fi

wc -l $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.zh

if [ -f $orig/WikiMatrix.v1.en-zh.langid.tsv ]
then
    echo "Adding Wiki Matrix"
    awk -F "\t" '{ if ($4 == "en" && $5 == "zh") {print $2}}' $orig/WikiMatrix.v1.en-zh.langid.tsv >> $OUTDIR/parallel/train.$lang.en
    awk -F "\t" '{ if ($4 == "en" && $5 == "zh") {print $3}}' $orig/WikiMatrix.v1.en-zh.langid.tsv >> $OUTDIR/parallel/train.$lang.zh
fi

wc -l $OUTDIR/parallel/train.$lang.en
wc -l $OUTDIR/parallel/train.$lang.zh

echo "Fetching Validation data $lang" 
sacrebleu -t wmt18 -l $lang --echo src > ${OUTDIR}/parallel/wmt18-$lang.src
sacrebleu -t wmt18 -l $lang --echo ref > ${OUTDIR}/parallel/wmt18-$lang.ref

sacrebleu -t wmt19 -l $lang --echo src > ${OUTDIR}/parallel/wmt19-$lang.src
sacrebleu -t wmt19 -l $lang --echo ref > ${OUTDIR}/parallel/wmt19-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt20 -l $lang --echo src > ${OUTDIR}/parallel/wmt20-$lang.src
sacrebleu -t wmt20 -l $lang --echo ref > ${OUTDIR}/parallel/wmt20-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt18 -l $rev_lang --echo src > ${OUTDIR}/parallel/wmt18-$rev_lang.src
sacrebleu -t wmt18 -l $rev_lang --echo ref > ${OUTDIR}/parallel/wmt18-$rev_lang.ref

sacrebleu -t wmt19 -l $rev_lang --echo src > ${OUTDIR}/parallel/wmt19-$rev_lang.src
sacrebleu -t wmt19 -l $rev_lang --echo ref > ${OUTDIR}/parallel/wmt19-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt20 -l $rev_lang --echo src > ${OUTDIR}/parallel/wmt20-$rev_lang.src
sacrebleu -t wmt20 -l $rev_lang --echo ref > ${OUTDIR}/parallel/wmt20-$rev_lang.ref

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

echo "Normalizing text to traditional chinese"
python $script_dir/traditional_to_simplified.py $OUTDIR/parallel/train.$lang.zh > $OUTDIR/parallel/train.$lang.sim.zh

for t in wmt18 wmt19 wmt20; do
    python $script_dir/traditional_to_simplified.py ${OUTDIR}/parallel/$t-$lang.ref > ${OUTDIR}/parallel/$t-$lang.sim.ref
    python $script_dir/traditional_to_simplified.py ${OUTDIR}/parallel/$t-$rev_lang.src > ${OUTDIR}/parallel/$t-$rev_lang.sim.src
done

# Hacky symlink to get same prefix for En and Zh
ln -s ${OUTDIR}/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.seg.en

echo "Filtering data based on max length and length ratio ..."
$moses_path/scripts/training/clean-corpus-n.perl \
    -ratio 1000 \
    ${OUTDIR}/parallel/train.$lang.sim \
    $lang1 $lang2 \
    ${OUTDIR}/parallel/train.$lang.filter \
    1 150

echo "Applying language ID filters"
fasttext predict $langid_model_path \
    ${OUTDIR}/parallel/train.$lang.filter.en \
    > ${OUTDIR}/parallel/train.$lang.filter.en.langid

fasttext predict $langid_model_path \
    ${OUTDIR}/parallel/train.$lang.filter.zh \
    > ${OUTDIR}/parallel/train.$lang.filter.zh.langid

paste -d "\t" \
    $OUTDIR/parallel/train.$lang.filter.en \
    $OUTDIR/parallel/train.$lang.filter.zh \
    $OUTDIR/parallel/train.$lang.filter.en.langid \
    $OUTDIR/parallel/train.$lang.filter.zh.langid \
    | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__zh") {print $1}}' > $OUTDIR/parallel/train.$lang.all.en

paste -d "\t" \
    $OUTDIR/parallel/train.$lang.filter.en \
    $OUTDIR/parallel/train.$lang.filter.zh \
    $OUTDIR/parallel/train.$lang.filter.en.langid \
    $OUTDIR/parallel/train.$lang.filter.zh.langid \
    | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__zh") {print $2}}' > $OUTDIR/parallel/train.$lang.all.zh

echo "Segmenting chinese ..."
python $script_dir/segment_zh.py $OUTDIR/parallel/train.$lang.all.zh > $OUTDIR/parallel/train.$lang.all.seg.zh

echo "=================================================="
echo "========== Moses Punct Normalization ============="
echo "=================================================="

echo "Normalizing punct ..."
for t in all; do
    for l in $lang1; do
        cat $OUTDIR/parallel/train.$lang.$t.$l | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $l | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > $OUTDIR/parallel/train.clean.$lang.$t.$l
        cat $OUTDIR/parallel/train.clean.$lang.$t.$l | perl $moses_path/scripts/tokenizer/tokenizer.perl -l $l -no-escape -threads 20 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$l
    done
    cp $OUTDIR/parallel/train.$lang.$t.$lang2 $OUTDIR/parallel/train.clean.$lang.$t.$lang2
    cp $OUTDIR/parallel/train.$lang.$t.seg.$lang2 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2

    cat $OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.common
    cat $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.common
    cat $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.srctok.common

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.$lang2.shuffled

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.srctok.$lang2.shuffled
done

for t in wmt18 wmt19 wmt20; do
    cat ${OUTDIR}/parallel/$t-$lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.src
    cp ${OUTDIR}/parallel/$t-$lang.seg.ref ${OUTDIR}/parallel/$t-$lang.clean.tok.ref
    cat ${OUTDIR}/parallel/$t-$lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$lang.clean.tok.src

    cat ${OUTDIR}/parallel/$t-$rev_lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.ref
    cp ${OUTDIR}/parallel/$t-$rev_lang.seg.src ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.ref
done

wmt_dir=$1;
out_dir=$2;
script_dir=$(pwd)
bicleaner_model_path=$3;
bifixer_path=$4;
langid_model_path=$5;
moses_path=$6;

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
    awk -F "\t" '{ if ($4 == "en" && $5 == "zh") {print $3}}' $orig/WikiMatrix.v1.en-zh.langid.tsv >> $OUTDIR/parallel/train.$lang.zh
fi

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

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

echo "Normalizing text to traditional chinese"
python $script_dir/traditional_to_simplified.py $OUTDIR/parallel/train.$lang.zh > $OUTDIR/parallel/train.$lang.sim.zh

for t in wmt19 wmt20; do
    python $script_dir/traditional_to_simplified.py ${OUTDIR}/parallel/$t-$lang.ref > ${OUTDIR}/parallel/$t-$lang.sim.ref
    python $script_dir/traditional_to_simplified.py ${OUTDIR}/parallel/$t-$rev_lang.src > ${OUTDIR}/parallel/$t-$rev_lang.sim.src
done

echo "Segmenting chinese ..."
python $script_dir/segment_zh.py $OUTDIR/parallel/train.$lang.sim.zh > $OUTDIR/parallel/train.$lang.seg.zh

for t in wmt19 wmt20; do
    python $script_dir/segment_zh.py ${OUTDIR}/parallel/$t-$lang.sim.ref > ${OUTDIR}/parallel/$t-$lang.seg.ref
    python $script_dir/segment_zh.py ${OUTDIR}/parallel/$t-$rev_lang.sim.src > ${OUTDIR}/parallel/$t-$rev_lang.seg.src
done

# Hacky symlink to get same prefix for En and Zh
ln -s ${OUTDIR}/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.seg.en

echo "Filtering data based on max length and length ratio ..."
$moses_path/scripts/training/clean-corpus-n.perl \
    -ratio 1.7 \
    ${OUTDIR}/parallel/train.$lang.seg \
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

echo "Lang ID and Bi-Cleaner"
paste -d "\t" \
    ${OUTDIR}/parallel/train.$lang.filter.en \
    ${OUTDIR}/parallel/train.$lang.filter.zh \
    ${OUTDIR}/parallel/train.$lang.filter.en.langid \
    ${OUTDIR}/parallel/train.$lang.filter.zh.langid \
    | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__zh") {print "-\t-\t"$1"\t"$2}}' \
    | bicleaner-classify - - $bicleaner_model_path > $OUTDIR/parallel/train.$lang.bicleaner.score

echo "Applying bifixer & dedup"
cat $OUTDIR/parallel/train.$lang.bicleaner.score \
| parallel -j 19 --pipe -k -l 30000 python $bifixer_path/bifixer.py \
    --ignore_segmentation -q - - en zh \
    | awk -F "\t" '!seen[$6]++' - > $OUTDIR/parallel/train.$lang.bifixer.score

echo "Creating data with different classifier confidence values ..."
awk -F "\t" '{print $3}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.en
awk -F "\t" '{print $4}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.zh

awk -F "\t" '{ if ($5>0.5) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.en
awk -F "\t" '{ if ($5>0.5) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.zh

awk -F "\t" '{ if ($5>0.6) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.en
awk -F "\t" '{ if ($5>0.6) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.zh

awk -F "\t" '{ if ($5>0.75) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.en
awk -F "\t" '{ if ($5>0.75) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.zh

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

    cat $OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.common
    cat $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.common

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.$lang2.shuffled

    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1.shuffled
    shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2.shuffled
done

for t in wmt19 wmt20; do
    cat ${OUTDIR}/parallel/$t-$lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.src
    cat ${OUTDIR}/parallel/$t-$lang.seg.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l zh | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l zh > ${OUTDIR}/parallel/$t-$lang.clean.tok.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.seg.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l zh | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l zh> ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.ref
done

echo "=================================================="
echo "========== Fetching Monolingual Data ============="
echo "=================================================="

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
    echo "Cleaning data ..."
    cat ${OUTDIR_MONO}/monolingual.news.dedup.en | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR_MONO}/monolingual.news.dedup.clean.en
    cat ${OUTDIR_MONO}/monolingual.news.dedup.clean.en | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR_MONO}/monolingual.news.dedup.clean.tok.en
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

if [ -f ${OUTDIR_MONO}/monolingual.news.dedup.seg.zh ]; then
    echo "Found segmented monolingual chinese data, skipping segmentation"
else
    echo "Segmenting monolingual chinese data ..."
    python $script_dir/traditional_to_simplified.py ${OUTDIR_MONO}/monolingual.news.dedup.zh > ${OUTDIR_MONO}/monolingual.news.dedup.sim.zh
    python $script_dir/segment_zh.py ${OUTDIR_MONO}/monolingual.news.dedup.sim.zh > ${OUTDIR_MONO}/monolingual.news.dedup.seg.zh
    cat ${OUTDIR_MONO}/monolingual.news.dedup.seg.zh | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l zh | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR_MONO}/monolingual.news.dedup.clean.zh
    cat ${OUTDIR_MONO}/monolingual.news.dedup.clean.zh | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l zh > ${OUTDIR_MONO}/monolingual.news.dedup.clean.tok.zh
fi

wmt_dir=$1;
out_dir=$2;
script_dir=$(pwd)
langid_model_path=$3;
moses_path=$4;#

mkdir -p ${out_dir}
mkdir -p ${wmt_dir}
mkdir -p ${wmt_dir}/orig

URLS=(
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/bonus/en-ru.txt.gz"
    "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-ru.tsv.gz"
    "http://data.statmt.org/wikititles/v2/wikititles-v2.ru-en.tsv.gz"
    #"http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.en.gz"
    #"http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.ru.gz"
    #"http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.translatedto.en.gz"
    #"http://data.statmt.org/wmt20/translation-task/back-translation/ru-en/news.translatedto.ru.gz"
    "http://data.statmt.org/wmt20/translation-task/WikiMatrix/WikiMatrix.v1.en-ru.langid.tsv.gz"
)

FILES=(
    "training-parallel-commoncrawl.tgz"
    "en-ru.txt.gz"
    "news-commentary-v15.en-ru.tsv.gz"
    "wikititles-v2.ru-en.tsv.gz"
    #"news.en.gz"
    #"news.ru.gz"
    #"news.translatedto.en.gz"
    #"news.translatedto.ru.gz"
    "WikiMatrix.v1.en-ru.langid.tsv.gz"
)

OUTDIR=$out_dir
lang1=en
lang2=ru
lang=en-ru
rev_lang=ru-en
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

echo "Adding Commoncrawl"
cat $orig/commoncrawl.ru-en.ru >> $OUTDIR/parallel/train.$lang.ru
cat $orig/commoncrawl.ru-en.en >> $OUTDIR/parallel/train.$lang.en

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding News Commentary v15"
awk -F "\t" '{print $1}' $orig/news-commentary-v15.en-ru.tsv >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/news-commentary-v15.en-ru.tsv >> $OUTDIR/parallel/train.$lang.ru

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Paracrawl 7.1"
awk -F "\t" '{print $1}' $orig/en-ru.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/en-ru.txt >> $OUTDIR/parallel/train.$lang.ru

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Wiki titles"
awk -F "\t" '{print $1}' $orig/wikititles-v2.ru-en.tsv >> $OUTDIR/parallel/train.$lang.ru
awk -F "\t" '{print $2}' $orig/wikititles-v2.ru-en.tsv >> $OUTDIR/parallel/train.$lang.en

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Wiki Matrix"
awk -F "\t" '{ if ($4 == "en" && $5 == "ru") {print $2}}' $orig/WikiMatrix.v1.en-ru.langid.tsv >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{ if ($4 == "en" && $5 == "ru") {print $3}}' $orig/WikiMatrix.v1.en-ru.langid.tsv >> $OUTDIR/parallel/train.$lang.ru

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

if [ -f $orig/UNv1.0.6way.tar.gz.00 ]
then
    echo "Adding UN Parallel Corpus"
    cat $orig/UNv1.0.6way.tar.gz.* > $orig/UNv1.0.6way.tar.gz
    tar -xzf $orig/UNv1.0.6way.tar.gz
    for l in $lang1 $lang2; do
        cat $orig/6way/UNv1.0.6way.$l >> $OUTDIR/parallel/train.$lang.$l
    done
fi

wc -l $OUTDIR/parallel/train.$lang.ru
wc -l $OUTDIR/parallel/train.$lang.en

echo "Applying LangID filters ..."

fasttext predict $langid_model_path $OUTDIR/parallel/train.$lang.en > $OUTDIR/parallel/train.$lang.langid.en
fasttext predict $langid_model_path $OUTDIR/parallel/train.$lang.ru > $OUTDIR/parallel/train.$lang.langid.ru

paste -d "\t" $OUTDIR/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.ru $OUTDIR/parallel/train.$lang.langid.en $OUTDIR/parallel/train.$lang.langid.ru | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__ru") {print $1}}' > $OUTDIR/parallel/train.$lang.langidfilter.en
paste -d "\t" $OUTDIR/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.ru $OUTDIR/parallel/train.$lang.langid.en $OUTDIR/parallel/train.$lang.langid.ru | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__ru") {print $2}}' > $OUTDIR/parallel/train.$lang.langidfilter.ru

wc -l $OUTDIR/parallel/train.$lang.langidfilter.en
wc -l $OUTDIR/parallel/train.$lang.langidfilter.ru

echo "Fetching Validation data $lang" 
sacrebleu -t wmt13 -l $lang --echo src > ${OUTDIR}/parallel/newstest2013-$lang.src
sacrebleu -t wmt13 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2013-$lang.ref

sacrebleu -t wmt14 -l $lang --echo src > ${OUTDIR}/parallel/newstest2014-$lang.src
sacrebleu -t wmt14 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2014-$lang.ref

sacrebleu -t wmt18 -l $lang --echo src > ${OUTDIR}/parallel/newstest2018-$lang.src
sacrebleu -t wmt18 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2018-$lang.ref

sacrebleu -t wmt19 -l $lang --echo src > ${OUTDIR}/parallel/newstest2019-$lang.src
sacrebleu -t wmt19 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2019-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt20 -l $lang --echo src > ${OUTDIR}/parallel/newstest2020-$lang.src
sacrebleu -t wmt20 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2020-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt13 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2013-$rev_lang.src
sacrebleu -t wmt13 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2013-$rev_lang.ref

sacrebleu -t wmt14 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2014-$rev_lang.src
sacrebleu -t wmt14 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2014-$rev_lang.ref

sacrebleu -t wmt18 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2018-$rev_lang.src
sacrebleu -t wmt18 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2018-$rev_lang.ref

sacrebleu -t wmt19 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2019-$rev_lang.src
sacrebleu -t wmt19 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2019-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt20 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2020-$rev_lang.src
sacrebleu -t wmt20 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2020-$rev_lang.ref

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

echo "Filtering data based on max length and length ratio ..."
$moses_path/scripts/training/clean-corpus-n.perl -ratio 1.6 $OUTDIR/parallel/train.$lang.langidfilter $lang1 $lang2 ${OUTDIR}/parallel/train.$lang.filter 1 250

echo "=================================================="
echo "========== Moses Punct Normalization ============="
echo "=================================================="

echo "Normalizing punct & tokenizing ..."
t=all
for l in $lang1 $lang2; do
    cat $OUTDIR/parallel/train.$lang.filter.$l | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l $l | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > $OUTDIR/parallel/train.clean.$lang.$t.$l
    cat $OUTDIR/parallel/train.clean.$lang.$t.$l | perl $moses_path/scripts/tokenizer/tokenizer.perl -l $l -no-escape -threads 20 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$l
done

cat $OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.common
cat $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.common

shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.$lang1.shuffled
shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.$lang2.shuffled

shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1.shuffled
shuf --random-source=$OUTDIR/parallel/train.clean.$lang.$t.tok.$lang1 $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2 > $OUTDIR/parallel/train.clean.$lang.$t.tok.$lang2.shuffled

echo "Normalizing valid/test punct & tokenizing ..."

for t in newstest2013 newstest2014 newstest2018 newstest2019 newstest2020; do
    cat ${OUTDIR}/parallel/$t-$lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.src
    cat ${OUTDIR}/parallel/$t-$lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l ru | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l ru > ${OUTDIR}/parallel/$t-$lang.clean.tok.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l ru | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l ru > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.ref
done

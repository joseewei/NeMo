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
    "https://www.statmt.org/europarl/v7/es-en.tgz"
    "http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz"
    "https://s3.amazonaws.com/web-language-models/paracrawl/release7.1/en-es.txt.gz"
    "http://data.statmt.org/news-commentary/v15/training/news-commentary-v15.en-es.tsv.gz"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EESC2017.en-es.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/rapid2016.en-es.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/ecb2017.en-es.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/EMA2016.en-es.tmx.zip"
    "https://tilde-model.s3-eu-west-1.amazonaws.com/worldbank.en-es.tmx.zip"
    "https://dl.fbaipublicfiles.com/laser/WikiMatrix/v1/WikiMatrix.en-es.tsv.gz"
)

FILES=(
    "es-en.tgz"
    "training-parallel-commoncrawl.tgz"
    "en-es.txt.gz"
    "news-commentary-v15.en-fr.tsv.gz"
    "EESC2017.en-es.tmx.zip"
    "rapid2016.en-es.tmx.zip"
    "ecb2017.en-es.tmx.zip"
    "EMA2016.en-es.tmx.zip"
    "worldbank.en-es.tmx.zip"
    "WikiMatrix.en-es.tsv.gz"
)

OUTDIR=$out_dir
lang1=en
lang2=es
lang=en-es
rev_lang=es-en
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
cat $orig/europarl-v7.es-en.en >> $OUTDIR/parallel/train.$lang.en
cat $orig/europarl-v7.es-en.es >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Commoncrawl"
cat $orig/commoncrawl.es-en.es >> $OUTDIR/parallel/train.$lang.es
cat $orig/commoncrawl.es-en.en >> $OUTDIR/parallel/train.$lang.en

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding News Commentary v15"
awk -F "\t" '{print $1}' $orig/news-commentary-v15.en-es.tsv >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/news-commentary-v15.en-es.tsv >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Paracrawl 7.1"
awk -F "\t" '{print $1}' $orig/en-es.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/en-es.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding WikiMatrix v1"
python $laser_path/extract.py \
    --tsv $orig/WikiMatrix.en-es.tsv.gz \
    --bitext $orig/WikiMatrix.en-es.txt \
    --src-lang en \
    --trg-lang es \
    --threshold 1.04

fasttext predict $langid_model_path $orig/WikiMatrix.en-es.txt.en > $orig/WikiMatrix.en-es.txt.en.langid
fasttext predict $langid_model_path $orig/WikiMatrix.en-es.txt.es > $orig/WikiMatrix.en-es.txt.es.langid

paste -d "\t" $orig/WikiMatrix.en-es.txt.en $orig/WikiMatrix.en-es.txt.es $orig/WikiMatrix.en-es.txt.en.langid $orig/WikiMatrix.en-es.txt.es.langid | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__es") {print $1}}' >> $OUTDIR/parallel/train.$lang.en
paste -d "\t" $orig/WikiMatrix.en-es.txt.en $orig/WikiMatrix.en-es.txt.es $orig/WikiMatrix.en-es.txt.en.langid $orig/WikiMatrix.en-es.txt.es.langid | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__es") {print $2}}' >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding EESC2017"
python $script_dir/tmx2txt.py --codelist en,es $orig/EESC.en-es.tmx $orig/EESC.en-es.tmx.txt
awk -F "\t" '{print $1}' $orig/EESC.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EESC.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Rapid2019"
python $script_dir/tmx2txt.py --codelist en,es $orig/rapid2016.en-es.tmx $orig/rapid2016.en-es.tmx.txt
awk -F "\t" '{print $1}' $orig/rapid2016.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/rapid2016.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding ECB2017"
python $script_dir/tmx2txt.py --codelist en,es $orig/ecb2017.UNIQUE.en-es.tmx $orig/ecb2017.UNIQUE.en-es.tmx.txt
awk -F "\t" '{print $1}' $orig/ecb2017.UNIQUE.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/ecb2017.UNIQUE.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding EMEA2016"
python $script_dir/tmx2txt.py --codelist en,es $orig/EMEA2016.en-es.tmx $orig/EMEA2016.en-es.tmx.txt
awk -F "\t" '{print $1}' $orig/EMEA2016.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/EMEA2016.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Adding Worldbank data"
python $script_dir/tmx2txt.py --codelist en,es $orig/worldbank.UNIQUE.en-es.tmx $orig/worldbank.UNIQUE.en-es.tmx.txt
awk -F "\t" '{print $1}' $orig/worldbank.UNIQUE.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.en
awk -F "\t" '{print $2}' $orig/worldbank.UNIQUE.en-es.tmx.txt >> $OUTDIR/parallel/train.$lang.es

wc -l $OUTDIR/parallel/train.$lang.es
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

wc -l $OUTDIR/parallel/train.$lang.es
wc -l $OUTDIR/parallel/train.$lang.en

echo "Applying LangID filters ..."

fasttext predict $langid_model_path $OUTDIR/parallel/train.$lang.en > $OUTDIR/parallel/train.$lang.langid.en
fasttext predict $langid_model_path $OUTDIR/parallel/train.$lang.es > $OUTDIR/parallel/train.$lang.langid.es

paste -d "\t" $OUTDIR/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.es $OUTDIR/parallel/train.$lang.langid.en $OUTDIR/parallel/train.$lang.langid.es | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__es") {print $1}}' > $OUTDIR/parallel/train.$lang.langidfilter.en
paste -d "\t" $OUTDIR/parallel/train.$lang.en $OUTDIR/parallel/train.$lang.es $OUTDIR/parallel/train.$lang.langid.en $OUTDIR/parallel/train.$lang.langid.es | awk -F "\t" '{ if ($3 == "__label__en" && $4 == "__label__es") {print $2}}' > $OUTDIR/parallel/train.$lang.langidfilter.es

wc -l $OUTDIR/parallel/train.$lang.langidfilter.en
wc -l $OUTDIR/parallel/train.$lang.langidfilter.es

echo "Fetching Validation data $lang" 
sacrebleu -t wmt12 -l $lang --echo src > ${OUTDIR}/parallel/newstest2012-$lang.src
sacrebleu -t wmt12 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2012-$lang.ref

echo "Fetching Test data $lang" 
sacrebleu -t wmt13 -l $lang --echo src > ${OUTDIR}/parallel/newstest2013-$lang.src
sacrebleu -t wmt13 -l $lang --echo ref > ${OUTDIR}/parallel/newstest2013-$lang.ref

echo "Fetching Validation data $rev_lang" 
sacrebleu -t wmt12 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2012-$rev_lang.src
sacrebleu -t wmt12 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2012-$rev_lang.ref

echo "Fetching Test data $rev_lang" 
sacrebleu -t wmt13 -l $rev_lang --echo src > ${OUTDIR}/parallel/newstest2013-$rev_lang.src
sacrebleu -t wmt13 -l $rev_lang --echo ref > ${OUTDIR}/parallel/newstest2013-$rev_lang.ref

echo "=================================================="
echo "========= Filtering and Cleaning Data ============"
echo "=================================================="

echo "Filtering data based on max length and length ratio ..."
$moses_path/scripts/training/clean-corpus-n.perl -ratio 1.3 ${OUTDIR}/parallel/train.$lang $lang1 $lang2 ${OUTDIR}/parallel/train.$lang.filter 1 250

echo "Applying bi-cleaner classifier"
awk '{print "-\t-"}' $OUTDIR/parallel/train.$lang.filter.en \
| paste -d "\t" - $OUTDIR/parallel/train.$lang.filter.en $OUTDIR/parallel/train.$lang.filter.es \
| bicleaner-classify - - $bicleaner_model_path > $OUTDIR/parallel/train.$lang.bicleaner.score

echo "Applying bifixer & dedup"
cat $OUTDIR/parallel/train.$lang.bicleaner.score \
| parallel -j 19 --pipe -k -l 30000 python $bifixer_path/bifixer.py \
    --ignore_segmentation -q - - en es \
    | awk -F "\t" '!seen[$6]++' - > $OUTDIR/parallel/train.$lang.bifixer.score

echo "Creating data with different classifier confidence values ..."
awk -F "\t" '{print $3}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.en
awk -F "\t" '{print $4}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.all.es

awk -F "\t" '{ if ($5>0.5) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.en
awk -F "\t" '{ if ($5>0.5) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.50.es

awk -F "\t" '{ if ($5>0.6) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.en
awk -F "\t" '{ if ($5>0.6) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.60.es

awk -F "\t" '{ if ($5>0.75) {print $3}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.en
awk -F "\t" '{ if ($5>0.75) {print $4}}' $OUTDIR/parallel/train.$lang.bifixer.score > $OUTDIR/parallel/train.$lang.75.es

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

for t in newstest2012 newstest2013; do
    cat ${OUTDIR}/parallel/$t-$lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.src
    cat ${OUTDIR}/parallel/$t-$lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l fr | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l fr > ${OUTDIR}/parallel/$t-$lang.clean.tok.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.src | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l fr | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.ref | perl $moses_path/scripts/tokenizer/normalize-punctuation.perl -l en | perl $moses_path/scripts/tokenizer/remove-non-printing-char.perl > ${OUTDIR}/parallel/$t-$rev_lang.clean.ref

    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.src | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l fr > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.src
    cat ${OUTDIR}/parallel/$t-$rev_lang.clean.ref | perl $moses_path/scripts/tokenizer/tokenizer.perl -threads 20 -no-escape -l en > ${OUTDIR}/parallel/$t-$rev_lang.clean.tok.ref
done

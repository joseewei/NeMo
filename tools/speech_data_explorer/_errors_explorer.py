# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import base64
import io
import json
import math
import operator
from collections import Counter, defaultdict

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import diff_match_patch
import edit_distance
import editdistance
import librosa
import numpy as np
import soundfile as sf
import tqdm
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
from plotly import express as px
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# number of items in a table per page
DATA_PAGE_SIZE = 10

# operators for filtering items
filter_operators = [
    ['ge ', '>='],
    ['le ', '<='],
    ['lt ', '<'],
    ['gt ', '>'],
    ['ne ', '!='],
    ['eq ', '='],
    ['contains '],
]

# parse table filter queries
def split_filter_part(filter_part):
    for operator_type in filter_operators:
        for op in operator_type:
            if op in filter_part:
                name_part, value_part = filter_part.split(op, 1)
                name = name_part[name_part.find('{') + 1 : name_part.rfind('}')]
                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1:-1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part
                return name, operator_type[0].strip(), value
    return [None] * 3


# standard command-line arguments parser
def parse_args():
    parser = argparse.ArgumentParser(description='ASR Errors Explorer')
    parser.add_argument(
        'manifest', help='path to JSON manifest file',
    )
    parser.add_argument('--port', default='8050', help='serving port for establishing connection')
    parser.add_argument('--debug', '-d', action='store_true', help='enable debug mode')
    args = parser.parse_args()
    return args


# load data from JSON manifest file
def load_data(data_filename):
    data = []
    wer_dist = 0.0
    wer_count = 0
    cer_dist = 0.0
    cer_count = 0
    vocabulary = defaultdict(lambda: 0)
    alphabet = set()

    orig_vocab = Counter()
    pred_vocab = defaultdict(lambda: 0)

    sm = edit_distance.SequenceMatcher()

    with open(data_filename, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            item = json.loads(line)
            num_words = len(item['text'].split())
            num_chars = len(item['text'])
            word_dist = editdistance.eval(item['text'].split(), item['transcript'].split())
            char_dist = editdistance.eval(item['text'], item['transcript'])
            wer_dist += word_dist
            cer_dist += char_dist
            wer_count += num_words
            cer_count += num_chars

            for word in item['text'].split():
                vocabulary[word] += 1
            for char in item['text']:
                alphabet.add(char)

            orig = item['text'].split()
            orig_vocab.update(orig)
            pred = item['transcript'].split()
            pred_vocab_cur = defaultdict(lambda: 0)
            sm.set_seqs(orig, pred)
            matches = [m for m in sm.get_matching_blocks()]
            assert len(set([m[0] for m in matches])) == len(matches)
            for m in matches:
                assert m[2] == 1
                pred_vocab_cur[orig[m[0]]] += 1
            for w in pred_vocab_cur:
                pred_vocab[w] += pred_vocab_cur[w]

            data.append(
                {
                    'audio_filepath': item['audio_filepath'],
                    'duration': round(item['duration'], 2),
                    'num_words': num_words,
                    'num_chars': num_chars,
                    'word_rate': round(num_words / item['duration'], 2),
                    'char_rate': round(num_chars / item['duration'], 2),
                    'WER': round(word_dist / num_words * 100.0, 2),
                    'CER': round(char_dist / num_chars * 100.0, 2),
                    'M/Nr': round(len(matches) / len(orig) * 100.0, 2),
                    'M/Np': round(len(matches) / len(pred) * 100.0, 2),
                    'text': item['text'],
                    'transcript': item['transcript'],
                }
            )

            for k in item:
                if k not in data[-1]:
                    data[-1][k] = item[k]

    wer = wer_dist / wer_count * 100.0
    cer = cer_dist / cer_count * 100.0

    acc_vocab = {}
    for w in orig_vocab:
        acc_vocab[w] = pred_vocab[w] / orig_vocab[w] * 100.0

    vocabulary_data = [
        {'word': word, 'count': vocabulary[word], 'accuracy': round(acc_vocab[word], 1)} for word in vocabulary
    ]
    return data, wer, cer, vocabulary_data, alphabet


# plot histogram of specified field in data list
def plot_histogram(data, key, label):
    fig = px.histogram(
        data_frame=[item[key] for item in data],
        nbins=50,
        log_y=True,
        labels={'value': label},
        opacity=0.5,
        color_discrete_sequence=['green'],
        height=200,
    )
    fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0))
    return fig


def plot_word_accuracy(vocabulary_data):
    labels = ['Unrecognized', 'Sometimes recognized', 'Always recognized']
    counts = [0, 0, 0]
    for word in vocabulary_data:
        if word['accuracy'] == 0:
            counts[0] += 1
        elif word['accuracy'] < 100:
            counts[1] += 1
        else:
            counts[2] += 1
    colors = ['red', 'orange', 'green']

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=counts,
                marker_color=colors,
                text=['{:.2%}'.format(count / sum(counts)) for count in counts],
                textposition='auto',
            )
        ]
    )
    fig.update_layout(
        showlegend=False, margin=dict(l=0, r=0, t=0, b=0, pad=0), height=200, yaxis={'title_text': '#words'}
    )

    return fig


args = parse_args()
print('Loading data...')
data, wer, cer, vocabulary, alphabet = load_data(args.manifest)
print('Starting server...')
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])

figure_wer = plot_histogram(data, 'WER', '#utterances')
figure_cer = plot_histogram(data, 'CER', '#utterances')
figure_word_acc = plot_word_accuracy(vocabulary)

stats_layout = [
    dbc.Row(dbc.Col(html.H5(children='Global Statistics'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        [
            dbc.Col(html.Div('Overall WER, %', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Overall CER, %', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Vocabulary size', className='text-secondary'), width=3, className='border-right'),
            dbc.Col(html.Div('Alphabet size', className='text-secondary'), width=3),
        ],
        className='bg-light mt-2 rounded-top border-top border-left border-right',
    ),
    dbc.Row(
        [
            dbc.Col(
                html.H5('{:.2f}'.format(wer), className='text-center p-1', style={'color': 'green', 'opacity': 0.7},),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5('{:.2f}'.format(cer), className='text-center p-1', style={'color': 'green', 'opacity': 0.7}),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5(
                    '{} words'.format(len(vocabulary)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
                className='border-right',
            ),
            dbc.Col(
                html.H5(
                    '{} chars'.format(len(alphabet)),
                    className='text-center p-1',
                    style={'color': 'green', 'opacity': 0.7},
                ),
                width=3,
            ),
        ],
        className='bg-light rounded-bottom border-bottom border-left border-right',
    ),
    dbc.Row(dbc.Col(html.H5(children='Alphabet'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        dbc.Col(html.Div('{}'.format(sorted(alphabet))),), className='mt-2 bg-light text-monospace rounded border'
    ),
    dbc.Row(dbc.Col(html.H5('WER (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='wer-graph', figure=figure_wer),),),
    dbc.Row(dbc.Col(html.H5('CER (per utterance)'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='cer-graph', figure=figure_cer),),),
    dbc.Row(dbc.Col(html.H5('Word accuracy distribution'), className='text-secondary'), className='mt-3'),
    dbc.Row(dbc.Col(dcc.Graph(id='word-acc-graph', figure=figure_word_acc),),),
    dbc.Row(dbc.Col(html.H5('Vocabulary'), className='text-secondary'), className='mt-3'),
    dbc.Row(
        dbc.Col(
            dash_table.DataTable(
                id='wordstable',
                columns=[
                    {'name': 'Word', 'id': 'word'},
                    {'name': 'Count', 'id': 'count'},
                    {'name': 'Accuracy, %', 'id': 'accuracy'},
                ],
                filter_action='custom',
                filter_query='',
                sort_action='custom',
                sort_mode='single',
                page_action='custom',
                page_current=0,
                page_size=DATA_PAGE_SIZE,
                cell_selectable=False,
                page_count=math.ceil(len(vocabulary) / DATA_PAGE_SIZE),
                sort_by=[{'column_id': 'word', 'direction': 'asc'}],
                style_cell={'maxWidth': 0, 'textAlign': 'left'},
                style_header={'color': 'text-primary'},
            ),
        ),
        className='m-2',
    ),
]


@app.callback(
    [Output('wordstable', 'data'), Output('wordstable', 'page_count')],
    [Input('wordstable', 'page_current'), Input('wordstable', 'sort_by'), Input('wordstable', 'filter_query')],
)
def update_wordstable(page_current, sort_by, filter_query):
    vocabulary_view = vocabulary
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            vocabulary_view = [x for x in vocabulary_view if getattr(operator, op)(x[col_name], filter_value)]
            print(len(vocabulary_view), len(vocabulary))
        elif op == 'contains':
            vocabulary_view = [x for x in vocabulary_view if filter_value in str(x[col_name])]
            print(len(vocabulary_view), len(vocabulary))

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        vocabulary_view = sorted(vocabulary_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(vocabulary_view):
        page_current = len(vocabulary_view) // DATA_PAGE_SIZE
    return [
        vocabulary_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(vocabulary_view) / DATA_PAGE_SIZE),
    ]


samples_layout = (
    [
        dbc.Row(dbc.Col(html.H5('Data'), className='text-secondary'), className='mt-3'),
        dbc.Row(
            dbc.Col(
                dash_table.DataTable(
                    id='datatable',
                    columns=[{'name': k.replace('_', ' '), 'id': k} for k in data[0]],
                    filter_action='custom',
                    filter_query='',
                    sort_action='custom',
                    sort_mode='single',
                    sort_by=[],
                    row_selectable='single',
                    selected_rows=[0],
                    page_action='custom',
                    page_current=0,
                    page_size=DATA_PAGE_SIZE,
                    page_count=math.ceil(len(data) / DATA_PAGE_SIZE),
                    style_cell={'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': 0, 'textAlign': 'left'},
                    style_header={'color': 'text-primary', 'text_align': 'center',},
                    style_cell_conditional=[{'if': {'column_id': 'audio_filepath'}, 'width': '15%'}]
                    + [
                        {'if': {'column_id': c}, 'width': '10%', 'text_align': 'center'}
                        for c in ['duration', 'num_words', 'num_chars', 'word_rate', 'char_rate']
                    ],
                ),
            )
        ),
    ]
    + [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(children=k.replace('_', ' ')),
                    width=2,
                    className='mt-1 bg-light text-monospace text-break small rounded border',
                ),
                dbc.Col(
                    html.Div(id='_' + k), className='mt-1 bg-light text-monospace text-break small rounded border'
                ),
            ]
        )
        for k in data[0]
    ]
    + [
        dbc.Row(
            [
                dbc.Col(
                    html.Div(children='diff'),
                    width=2,
                    className='mt-1 bg-light text-monospace text-break small rounded border',
                ),
                dbc.Col(
                    html.Iframe(
                        # enable all sandbox features
                        # see https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe
                        # this prevents javascript from running inside the iframe
                        # and other things security reasons
                        id='_diff',
                        sandbox='',
                        srcDoc='',
                        style={'border': 'none', 'width': '100%', 'height': '100%'},
                        className='bg-light text-monospace text-break small',
                    ),
                    className='mt-1 bg-light text-monospace text-break small rounded border'
                    #               dcc.Markdown(id='_diff', dangerously_allow_html=True)  #, className='mt-1 bg-light text-monospace text-break small rounded border'
                ),
            ]
        )
    ]
    + [
        dbc.Row(dbc.Col(html.Audio(id='player', controls=True),), className='mt-3 '),
        dbc.Row(dbc.Col(dcc.Graph(id='signal-graph')), className='mt-3'),
    ]
)


@app.callback(
    [Output('datatable', 'data'), Output('datatable', 'page_count')],
    [Input('datatable', 'page_current'), Input('datatable', 'sort_by'), Input('datatable', 'filter_query')],
)
def update_datatable(page_current, sort_by, filter_query):
    data_view = data
    filtering_expressions = filter_query.split(' && ')
    for filter_part in filtering_expressions:
        col_name, op, filter_value = split_filter_part(filter_part)

        if op in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            data_view = [x for x in data_view if getattr(operator, op)(x[col_name], filter_value)]
        elif op == 'contains':
            data_view = [x for x in data_view if filter_value in str(x[col_name])]

    if len(sort_by):
        col = sort_by[0]['column_id']
        descending = sort_by[0]['direction'] == 'desc'
        data_view = sorted(data_view, key=lambda x: x[col], reverse=descending)
    if page_current * DATA_PAGE_SIZE >= len(data_view):
        page_current = len(data_view) // DATA_PAGE_SIZE
    return [
        data_view[page_current * DATA_PAGE_SIZE : (page_current + 1) * DATA_PAGE_SIZE],
        math.ceil(len(data_view) / DATA_PAGE_SIZE),
    ]


app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=False),
        dbc.NavbarSimple(
            children=[
                dbc.NavItem(dbc.NavLink('Statistics', id='stats_link', href='/', active=True)),
                dbc.NavItem(dbc.NavLink('Samples', id='samples_link', href='/samples')),
            ],
            brand='ASR Errors Explorer',
            sticky='top',
            color='green',
            dark=True,
        ),
        dbc.Container(id='page-content'),
    ]
)


@app.callback(
    [Output('page-content', 'children'), Output('stats_link', 'active'), Output('samples_link', 'active')],
    [Input('url', 'pathname')],
)
def nav_click(url):
    if url == '/samples':
        return [samples_layout, False, True]
    else:
        return [stats_layout, True, False]


@app.callback(
    [Output('_' + k, 'children') for k in data[0]], [Input('datatable', 'selected_rows'), Input('datatable', 'data')]
)
def show_item(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    return [data[idx[0]][k] for k in data[0]]


@app.callback(Output('_diff', 'srcDoc'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def show_diff(idx, data, orig='text_no_oov', pred='transcript'):
    if len(idx) == 0:
        raise PreventUpdate

    orig_words = data[idx[0]][orig]
    while '  ' in orig_words:
        orig_words = orig_words.replace('  ', ' ')
    orig_words = orig_words.replace(' ', '\n') + '\n'

    pred_words = data[idx[0]][pred]
    while '  ' in pred_words:
        pred_words = pred_words.replace('  ', ' ')
    pred_words = pred_words.replace(' ', '\n') + '\n'

    diff = diff_match_patch.diff_match_patch()
    diff.Diff_Timeout = 0
    orig_enc, pred_enc, enc = diff.diff_linesToChars(orig_words, pred_words)
    diffs = diff.diff_main(orig_enc, pred_enc, False)
    diff.diff_charsToLines(diffs, enc)
    diffs_post = []
    for d in diffs:
        diffs_post.append((d[0], d[1].replace('\n', ' ')))

    diff_html = diff.diff_prettyHtml(diffs_post)
    print(diff_html)

    return diff_html


@app.callback(Output('signal-graph', 'figure'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def plot_signal(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    figs = make_subplots(rows=2, cols=1, subplot_titles=('Waveform', 'Spectrogram'))
    try:
        filename = data[idx[0]]['audio_filepath']
        audio, fs = librosa.load(filename, sr=None)
        time_stride = 0.01
        hop_length = int(fs * time_stride)
        n_fft = 512
        # linear scale spectrogram
        s = librosa.stft(y=audio, n_fft=n_fft, hop_length=hop_length)
        s_db = librosa.power_to_db(np.abs(s) ** 2, ref=np.max, top_db=100)
        figs.add_trace(
            go.Scatter(
                x=np.arange(audio.shape[0]) / fs,
                y=audio,
                line={'color': 'green'},
                name='Waveform',
                hovertemplate='Time: %{x:.2f} s<br>Amplitude: %{y:.2f}<br><extra></extra>',
            ),
            row=1,
            col=1,
        )
        figs.add_trace(
            go.Heatmap(
                z=s_db,
                colorscale=[[0, 'rgb(30,62,62)'], [0.5, 'rgb(30,128,128)'], [1, 'rgb(30,255,30)'],],
                colorbar=dict(yanchor='middle', lenmode='fraction', y=0.2, len=0.5, ticksuffix=' dB'),
                dx=time_stride,
                dy=fs / n_fft / 1000,
                name='Spectrogram',
                hovertemplate='Time: %{x:.2f} s<br>Frequency: %{y:.2f} kHz<br>Magnitude: %{z:.2f} dB<extra></extra>',
            ),
            row=2,
            col=1,
        )
        figs.update_layout({'margin': dict(l=0, r=0, t=20, b=0, pad=0), 'height': 500})
        figs.update_xaxes(title_text='Time, s', row=1, col=1)
        figs.update_yaxes(title_text='Amplitude', row=1, col=1)
        figs.update_xaxes(title_text='Time, s', row=2, col=1)
        figs.update_yaxes(title_text='Frequency, kHz', row=2, col=1)
    except Exception:
        pass

    return figs


@app.callback(Output('player', 'src'), [Input('datatable', 'selected_rows'), Input('datatable', 'data')])
def update_player(idx, data):
    if len(idx) == 0:
        raise PreventUpdate
    try:
        filename = data[idx[0]]['audio_filepath']
        signal, sr = librosa.load(filename, sr=None)
        with io.BytesIO() as buf:
            sf.write(buf, signal, sr, format='WAV')
            buf.seek(0)
            encoded = base64.b64encode(buf.read())
        # print('\nDEBUG: {}\n'.format(len(encoded)))
        encoded = base64.b64encode(open(filename, 'rb').read())
        return 'data:audio/wav;base64,{}'.format(encoded.decode())
    except Exception:
        return ''


if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=args.port, debug=args.debug)

# Import Streamlit and Pandas
import math
import os
import random

import streamlit as st
import pandas as pd

# Import for API calls
import requests

# Import for navbar
from streamlit_option_menu import option_menu

# Import for dyanmic tagging
from streamlit_tags import st_tags, st_tags_sidebar

# Imports for aggrid
from st_aggrid import AgGrid
import pandas as pd
from data import chatlist

# The code below is for the title and logo.
st.set_page_config(layout='wide', page_title="Zero-Shot Text Classifier", page_icon="assets/logo.png")

colors = [
    "#ff2b2b",
    "#ff8700",
    "#ffd16a",
    "#0068c9",
    "#29b09d",
    "#83c9ff",
    "#ff2b2b",
    "#0068c9",
    "#29b09d",
    "#ff8700",
    "#6d3fc0",
    "#ffd16a",
    "#83c9ff",
    "#ffabab",
    "#7defa1",
    "#a181de",
    "#d5dae5",
];


def main():
    st.caption("")


col1, col2 = st.columns([2, 3])
with col1:
    c1, c2 = st.columns([0.4, 2])
    with c1:
        st.image("assets/logo.png", width=100, )
    with c2:
        st.caption("")
        st.title("零样本分类")
    st.write("")
    st.markdown(
        """
        使用这款应用程序快速、即时地对关键词进行分类。无需机器学习培训！
    """
    )

    option = st.selectbox('选择样例数据', [x[0] for x in chatlist], key='option2')
    selected = list(filter(lambda x: x[0] == option, chatlist))
    selected = selected[0]
    template = '\n'.join(selected)
    st.session_state.template = template
    with st.form(key="my_form"):
        API_KEY2 = st.text_input(
            "输入你的 🤗 HuggingFace API 令牌",
            help="创建 HuggiginFace 帐户后，您可以在设置页面中获取免费的 API 令牌：: https://huggingface.co/settings/tokens",
            type='password',
            value=os.getenv('HF_API_KEY')
        )

        API_URL = ("https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-9")
        # API_URL = ("https://api-inference.huggingface.co/models/finiteautomata/bertweet-base-sentiment-analysis")

        headers = {"Authorization": f"Bearer {API_KEY2}"}

        multiselectComponent = st_tags(
            label="输入分类词：",
            text="Add labels - 6 max",
            value=[
                'Positive', 'Negative', 'Neutral',
            ],
            # value=[
            #     'appreciation',
            #     'excitement',
            #     'expectation',
            #     'dissatisfaction',
            #     'disappointment',
            #      ],

            suggestions=[
                'appreciation',
                'excitement',
                'expectation',
                'dissatisfaction',
                'disappointment',
                'anger',
                "信息",
                "交易",
                "正面",
                "负面",
                "中性",
                'Positive', 'Negative', 'Neutral',
                'Happy', 'Angry'
            ],
            maxtags=6,
        )

        linesDeduped2 = []

        if option:
            MAX_LINES_FULL = 50
            text = st.text_area(
                "Enter keyphrases to classify",
                st.session_state.template,
                height=200,
                key="2",
                help="At least two keyphrases for the classifier to work, one per line, "
                     + str(MAX_LINES_FULL)
                     + " keyphrases max in 'unlocked mode'. You can tweak 'MAX_LINES_FULL' in the code to change this",
            )

            lines = text.split("\n")  # A list of lines
            linesList = []
            for x in lines:
                linesList.append(x.strip())
            linesList = list(dict.fromkeys(linesList))  # Remove dupes from list
            linesList = list(filter(None, linesList))  # Remove empty lines from list

            if len(linesList) > MAX_LINES_FULL:
                st.info(
                    f"❄️ Note that only the first "
                    + str(MAX_LINES_FULL)
                    + " keyprases will be reviewed to preserve performance. Fork the repo and tweak 'MAX_LINES_FULL' in the code to increase that limit."
                )

                linesList = linesList[:MAX_LINES_FULL]

            submit_button = st.form_submit_button(label="提交", type='primary')

if not "valid_inputs_received" in st.session_state:
    st.session_state["valid_inputs_received"] = False
if not submit_button and not st.session_state.valid_inputs_received:
    st.stop()

elif submit_button and not text:
    st.warning("❄️ There is no keyphrases to classify")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button and not multiselectComponent:
    st.warning("❄️ You have not added any labels, please add some! ")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button and len(multiselectComponent) == 1:
    st.warning("❄️ Please make sure to add at least two labels for classification")
    st.session_state.valid_inputs_received = False
    st.stop()

elif submit_button or st.session_state.valid_inputs_received:
    with col2:
        try:
            if submit_button:
                st.session_state.valid_inputs_received = True


            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()


            listToAppend = []

            for row in linesList:
                output2 = query({
                    "inputs": row,
                    "parameters": {"candidate_labels": multiselectComponent},
                    "options": {"wait_for_model": True},
                })
                import copy

                jsonresoult = copy.deepcopy(output2)
                jsonresoult = {'sequence': output2['sequence']}
                for (i, label) in enumerate(output2['labels']):
                    jsonresoult[label] = output2['scores'][i]
                listToAppend.append(jsonresoult)
                df = pd.DataFrame.from_dict(output2)
            st.success("✅ Done!")

            bubbledate = []
            for index, row in enumerate(listToAppend):
                for label in multiselectComponent:
                    bubbledate.append({
                        'score': row[label] * 50,
                        # 'score': math.sqrt(row[label] * 50),
                        'score_org': row[label],
                        '句子': row['sequence'],
                        'index': index + 1,
                        'label': label,
                    })
            print('-' * 80)
            print(bubbledate)
            df_bubble = pd.DataFrame.from_dict(bubbledate)
            print(df_bubble)

            print('*' * 80)
            print(listToAppend)

            df = pd.DataFrame.from_dict(listToAppend)
            print(df)

            # This code is to rename the columns
            df.rename(columns={"sequence": "句子"}, inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                st.write(df)
                st.write(df.mean(numeric_only=True))

            with col2:
                import plotly.express as px

                fig = px.scatter(
                    df_bubble,
                    x='index',
                    y='score',
                    size='score',
                    color="label",
                    log_x=True,
                    size_max=60,
                    color_discrete_sequence=colors,
                )
                st.plotly_chart(fig, theme="streamlit", use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.area_chart(df, x='句子', y=multiselectComponent, color=colors[:len(multiselectComponent)])
            with col2:
                st.bar_chart(df, x='句子', y=multiselectComponent, color=colors[:len(multiselectComponent)])


        except UnicodeError as ve:
            print(ve)
            st.warning("❄️ Add a valid HuggingFace API key in the text box above ☝️")
            st.stop()

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px


def plot_stacked_bar_chart2(top_10_drugs,disease):
    df = top_10_drugs.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

     # Ensure all 'Positive', 'Negative', and 'Neutral' categories are present
    for category in ['Positive', 'Negative', 'Neutral']:
        if category not in df.columns:
            df[category] = 0

    # Reset index to make 'drug' a separate column
    df = df.reset_index()
    #st.write(df)

    # Calculate total count for each drug
    df['Total'] = df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

    # Melt the DataFrame for Plotly
    df_melted = df.melt(id_vars=['drug', 'Total'], value_vars=['Positive', 'Negative', 'Neutral'], var_name='Sentiment',
                        value_name='Count')

    # Plot with Plotly Express
    fig = px.bar(df_melted, x='Count', y='drug', color='Sentiment', orientation='h',
                 height=800, width=1000, title='Sentiment Counts by Drug for {disease}',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'},
                 labels={'Count': 'Count per Sentiment'})

    # Add total count annotations
    for i, row in df.iterrows():
        fig.add_annotation(x=row['Total'], y=row['drug'], text=f'{row["Total"]}', showarrow=False, font=dict(color='black', size=12))

    # Update layout for better spacing and legend background color
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'},
                      legend=dict(bgcolor='black', bordercolor='black', borderwidth=1))

    # Display the plot
    st.plotly_chart(fig)



def plot_stacked_bar_chart_3(df, drug):
    # Filter the DataFrame to include only the top 10 drugs and the specified disease
    top_15_drugs_df = df[df['drug'] == drug]
    #get all records of top drugs from whole drug review datasets.
    #top_15_drugs_df = top_15_drugs_df[top_15_drugs_df['drug'].isin(top_10_drugs['drug'])]

    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = top_15_drugs_df.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Ensure all 'Positive', 'Negative', and 'Neutral' categories are present
    for category in ['Positive', 'Negative', 'Neutral']:
        if category not in grouped_df.columns:
            grouped_df[category] = 0

    # Reset index to make 'drug' a separate column
    grouped_df = grouped_df.reset_index()

    # Calculate total count for each drug
    grouped_df['Total'] = grouped_df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

    # Melt the DataFrame for Plotly
    df_melted = grouped_df.melt(id_vars=['drug', 'Total'], value_vars=['Positive', 'Negative', 'Neutral'],
                                var_name='Sentiment', value_name='Count')

    # Plot with Plotly Express (zoomable by default)
    fig = px.bar(df_melted, x='Count', y='drug', color='Sentiment', orientation='h',
                 height=800, width=1000, title=f'Review Sentiment Counts for {drug}',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'},
                 labels={'Count': 'Count per Sentiment'})

    # Add total count annotations at the end of each bar
    for i, row in grouped_df.iterrows():
        fig.add_annotation(x=row['Total'], y=row['drug'], text=f'{row["Total"]}', showarrow=False,
                           font=dict(color='black', size=12))

    # Update layout for better spacing and legend background color
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'},
                      legend=dict(bgcolor='white', bordercolor='black', borderwidth=1))

    # Show the zoomable plot
    # Display the plot in Streamlit
    st.plotly_chart(fig)  # Use st.plotly_chart instead of fig.show()



def plot_stacked_bar_chartavg(df, top_10_drugs, disease):
    # Filter the DataFrame to include only the top 10 drugs and the specified disease
    top_15_drugs_df = df[df['Disease'] == disease]
    top_15_drugs_df = top_15_drugs_df[top_15_drugs_df['drug'].isin(top_10_drugs['drug'])]

    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = top_15_drugs_df.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    #st.write("in plot_stacked_bar_chartavg")
    #st.write(grouped_df)
    # Ensure all 'Positive', 'Negative', and 'Neutral' categories are present
    for category in ['Positive', 'Negative', 'Neutral']:
        if category not in grouped_df.columns:
            grouped_df[category] = 0

    # Reset index to make 'drug' a separate column
    grouped_df = grouped_df.reset_index()

    # Calculate total count for each drug
    grouped_df['Total'] = grouped_df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

    st.write("in plot_stacked_bar_chartavg")
    st.write(grouped_df)

    # Melt the DataFrame for Plotly
    df_melted = grouped_df.melt(id_vars=['drug', 'Total'], value_vars=['Positive', 'Negative', 'Neutral'],
                                var_name='Sentiment', value_name='Count')

    # Plot with Plotly Express (zoomable by default)
    fig = px.bar(df_melted, x='Count', y='drug', color='Sentiment', orientation='h',
                 height=800, width=1000, title=f'Review Sentiment Counts by Drug for {disease}',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'},
                 labels={'Count': 'Count per Sentiment'})

    # Add total count annotations at the end of each bar
    for i, row in grouped_df.iterrows():
        fig.add_annotation(x=row['Total'], y=row['drug'], text=f'{row["Total"]}', showarrow=False,
                           font=dict(color='black', size=12))

    # Update layout for better spacing and legend background color
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'},
                      legend=dict(bgcolor='white', bordercolor='black', borderwidth=1))

    # Show the zoomable plot
    fig.show()


def plot_stacked_bar_chartavg2(df, top_10_drugs, disease):
    # Filter the DataFrame to include only the top 10 drugs and the specified disease
    top_15_drugs_df = df[df['Disease'] == disease]
    top_15_drugs_df = top_15_drugs_df[top_15_drugs_df['drug'].isin(top_10_drugs['drug'])]

    # Group by 'drug' and 'rating_category' and count occurrences
    grouped_df = top_15_drugs_df.groupby(['drug', 'rating_category']).size().unstack(fill_value=0)

    # Ensure all 'Positive', 'Negative', and 'Neutral' categories are present
    for category in ['Positive', 'Negative', 'Neutral']:
        if category not in grouped_df.columns:
            grouped_df[category] = 0

    # Reset index to make 'drug' a separate column
    grouped_df = grouped_df.reset_index()

    # Calculate total count for each drug
    grouped_df['Total'] = grouped_df[['Positive', 'Negative', 'Neutral']].sum(axis=1)

    #st.write("in plot_stacked_bar_chartavg")
    #st.write(grouped_df)

    # Melt the DataFrame for Plotly
    df_melted = grouped_df.melt(id_vars=['drug', 'Total'], value_vars=['Positive', 'Negative', 'Neutral'],
                                var_name='Sentiment', value_name='Count')

    # Plot with Plotly Express (zoomable by default)
    fig = px.bar(df_melted, x='Count', y='drug', color='Sentiment', orientation='h',
                 height=800, width=1000, title=f'Review Sentiment Counts by Drug for {disease}',
                 color_discrete_map={'Negative': 'red', 'Neutral': 'orange', 'Positive': 'green'},
                 labels={'Count': 'Count per Sentiment'})

    # Add total count annotations at the end of each bar
    for i, row in grouped_df.iterrows():
        fig.add_annotation(x=row['Total'], y=row['drug'], text=f'{row["Total"]}', showarrow=False,
                           font=dict(color='black', size=12))

    # Update layout for better spacing and legend background color
    fig.update_layout(barmode='stack', yaxis={'categoryorder': 'total ascending'},
                      legend=dict(bgcolor='white', bordercolor='black', borderwidth=1))

    # Display the plot in Streamlit
    st.plotly_chart(fig)  # Use st.plotly_chart instead of fig.show()


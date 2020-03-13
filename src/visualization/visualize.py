import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

ROOT = pathlib.Path().absolute().parent
RAW_DATA_PATH = ROOT / 'data' / 'raw'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'

class SalesVisualizations:
    
    def __init__(self):
        self.df = pd.read_pickle(PROCESSED_DATA_PATH / 'aggregated_dataset.pickle')

    def sales_by_store_and_state(self):
        
        fig = make_subplots(
            rows=3, cols=3,
            shared_yaxes=True,
            vertical_spacing=0.1,
            specs=[ [{'colspan':3},None,None],
                    [{'colspan':3},None,None],
                    [{},{},{}] ],
            subplot_titles=[
                'Total sales',
                'Sales by State per Store',
                'California','Texas','Wisconsin'])

        ##################################################
        ################## Total Sales ###################
        ##################################################
        
        # total aggregated sales
        to_plot = \
        (self.df
        .loc[:, 'sales']
        .sum(axis=1) # sum across departments
        .groupby('date')
        .sum() # sum across stores
        .asfreq('D') # specify frequency of index (there are no missing values)
        .rename('sales'))

        traces = [go.Scatter(
            x=to_plot.index,
            y=to_plot,
            name='Daily sales')]

        # also plot a smoother line of monthly sales
        # cannot use 'M' as that is not a fixed offset (not all months are same length)
        to_plot = \
        (to_plot
        .rolling('30D')
        .mean())

        traces.append(
            go.Scatter(
            x=to_plot.index,
            y=to_plot,
            name='30 Day Mean'))

        fig.add_traces(traces, rows=[1,1], cols=[1,1])

        ##################################################
        ################ Sales by State ##################
        ##################################################

        # compare sales across states and stores
        store_sales = \
        (self.df
        .loc[:, 'sales']
        .sum(axis=1) # sum across departments
        .droplevel(['year','month','weekday'])
        .unstack(['state_id','store_id']) # no missing values
        .asfreq('D')
        .resample('M')
        .sum())

        # sum stores within a state
        state_sales = \
        (store_sales
        .groupby(level='state_id', axis=1)
        .sum())

        # middle row is comparison of states
        traces = []
        for state_id in state_sales.columns:
            traces.append(go.Scatter(
                x=state_sales.index,
                y=state_sales[state_id] / {'WI':3,'TX':3,'CA':4}[state_id],
                name=state_id))

        fig.add_traces(traces, rows=[2,2,2], cols=[1,1,1])

        ##################################################
        ################ Sales by Store ##################
        ##################################################

        # bottom row compares stores in states
        for col, state_id in enumerate(state_sales.columns):
            traces = []
            tmp = store_sales[state_id]
            for store_id in tmp.columns:
                traces.append(go.Scatter(
                    x=tmp.index,
                    y=tmp[store_id],
                    name=store_id,))
            fig.add_traces(traces, rows=[3]*len(traces), cols=[col+1]*len(traces))

        layout = go.Layout(
            width=1000,
            height=800,
            margin=go.layout.Margin(l=10,t=50,b=10),
            yaxis =dict(title='Daily sales'),
            yaxis2=dict(title='Monthly sales'),
            yaxis3=dict(title='Monthly sales'))

        fig.update_layout(layout)
        fig.update_yaxes(rangemode='tozero')
        return fig

    def sales_per_category(self):
        
        fig = make_subplots(
            rows=2, cols=3,
            shared_yaxes=True,
            vertical_spacing=0.1,
            specs=[ [{'colspan':3},None,None],
                    # [{'colspan':3},None,None],
                    [{},{},{}] ],
            subplot_titles=[
                'Sales per Category',
                # 'Sales by State per Store',
                'Foods','Hobbies','Household'])

        # top row sales per category
        cat_sales = \
        (self.df
        ['sales']
        .groupby(level=0, axis=1) # group the categories
        .sum()
        .groupby(level='date')
        .sum()
        .resample('M')
        .sum())

        traces = []
        for cat_id in cat_sales.columns:
            traces.append(go.Scatter(
                x=cat_sales.index,
                y=cat_sales[cat_id] / {'FOODS':3,'HOBBIES':2,'HOUSEHOLD':2}[cat_id],
                name=cat_id))

        fig.add_traces(traces, rows=[1,1,1], cols=[1,1,1])

        # sales per department (3 columns)
        dept_sales = \
        (self.df
        ['sales']
        .groupby(level='date')
        .sum()
        .resample('M')
        .sum())

        # bottom row compares stores in states
        for col, cat_id in enumerate(cat_sales.columns):
            traces = []
            tmp = dept_sales[cat_id]
            for dept_id in tmp.columns:
                traces.append(go.Scatter(
                    x=tmp.index,
                    y=tmp[dept_id],
                    name=dept_id,))
            fig.add_traces(traces, rows=[2]*len(traces), cols=[col+1]*len(traces))

        layout = go.Layout(
            width=1000,
            height=500,
            margin=go.layout.Margin(l=10,t=50,b=10),
            yaxis =dict(title='Monthly sales'),
            yaxis2=dict(title='Monthly sales'))

        fig.update_layout(layout)
        fig.update_yaxes(rangemode='tozero')
        fig.for_each_xaxis(lambda x: x.update(matches='x2') if x.anchor != 'y' else x)
        return fig

    def sales_per_department_and_store(self):
        
        store_dept_sales = \
        (self.df
        ['sales']
        .droplevel(['year','month','weekday','state_id'])
        .unstack('store_id')
        .reorder_levels([2,0,1], axis='columns')
        .sort_index(axis=1)
        .resample('30D')
        .sum())

        store_ids = store_dept_sales.columns.get_level_values('store_id').unique().tolist()
        cat_ids = store_dept_sales.columns.get_level_values(level=1).unique().tolist()

        fig = make_subplots(rows=3, cols=10, shared_xaxes=True, shared_yaxes=True, column_titles=store_ids, row_titles=cat_ids)

        for col, store_id in enumerate(store_ids):
            for row, cat_id in enumerate(cat_ids):
                traces = []
                tmp = store_dept_sales.loc[:, (store_id, cat_id)]
                for i, dept_id in enumerate(tmp.columns):
                    traces.append(go.Scatter(
                        x=tmp.index,
                        y=tmp[dept_id],
                        name=dept_id,
                        legendgroup=dept_id,
                        showlegend=not col,
                        line_color=['blue','red','green'][i],
                        mode='lines',
                        xaxis=f'x{col+1}' if col else 'x',
                        yaxis=f'y{row+1}' if row else 'y',
                        ))
                fig.add_traces(traces, rows=[row+1]*len(traces), cols=[col+1]*len(traces))

        layout = go.Layout(
            title='Monthly sales by Department for Stores',
            legend=go.layout.Legend(
                orientation='h',
                y=-0.1, yanchor='top',
                x= 0.5, xanchor='center'),
            width=1200,
            height=600,
            margin=go.layout.Margin(l=10,t=100,b=50,r=10))

        fig.update_layout(layout)
        fig.update_xaxes(matches='x')
        fig.update_yaxes(rangemode='tozero')
        fig.for_each_xaxis(lambda x: x.update(showticklabels=True, tickangle=300) if x.anchor in [f'y{i}' for i in range(21,31)] else x)
        return fig
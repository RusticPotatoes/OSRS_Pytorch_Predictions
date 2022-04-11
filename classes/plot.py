import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


# dataset= data[['Close']] , xaxisticks=(range(0,data.shape[0],500),data['Date'].loc[::500]), title=("Amazon Stock Price"), xlabel='Date', ylabel='Close Price (USD)'
def plot_single(dataset, xaxisticks, xaxisdata, title, xlabel, ylabel):
	sns.set_style("darkgrid")
	fig = plt.figure(figsize=(15, 9))
	plt.plot(dataset)
	# fix ticks bug
	plt.xticks(xaxisticks, xaxisdata, rotation=330)
	plt.title(title, fontsize=18, fontweight='bold')
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(ylabel, fontsize=18)
	plt.show()
	return plt


def plot_dual(original, predict, hist, modelname, title, xlabel, ylabel):
	sns.set_style("darkgrid")	

	fig = plt.figure()
	fig.subplots_adjust(hspace=0.2, wspace=0.2)

	plt.subplot(1, 2, 1)
	ax = sns.lineplot(x=original.index, y=original[0], label="True History", color='royalblue')
	ax = sns.lineplot(x=predict.index, y=predict[0], label=f"Prediction ({modelname})", color='tomato')
	ax.set_title(title, size=14, fontweight='bold')
	ax.set_xlabel(xlabel, size=14)
	ax.set_ylabel(ylabel, size=14)
	ax.set_xticklabels('', size=10)

	plt.subplot(1, 2, 2)
	ax = sns.lineplot(data=hist, color='royalblue')
	ax.set_xlabel("Epoch", size=14)
	ax.set_ylabel("Loss", size=14)
	ax.set_title("Training Loss", size=14, fontweight='bold')
	fig.set_figheight(6)
	fig.set_figwidth(16)

	return fig 


def plot_data_df(df):
	data = []
	for col in df.columns:
		data.append(go.Scatter(x=df.index, y=df[col], name=col))
	fig = go.Figure(data=data)
	fig.show()


def plot_multi(result, title, feature):
	fig = go.Figure()
	fig.add_trace(
		go.Scatter(
			go.Scatter(
				x=result.index, y=result[0],
				mode='lines',
				name='Train'
				)
			)
		)
	fig.add_trace(
		go.Scatter(
			x=result.index, 
			y=result[1],
			mode='lines',
			name='Test'
			)
		)
	fig.add_trace(
		go.Scatter(
			x=result.index, 
			y=result[2],
			mode='lines',
			name='Forecast'
			)
		)
	fig.add_trace(
		go.Scatter(
			go.Scatter(
				x=result.index, 
				y=result[3],
				mode='lines',
				name='Actual'
				)
			)
		)
	fig.update_layout(
		xaxis=dict(
			showline=True,
			showgrid=True,
			showticklabels=False,
			linecolor='white',
			linewidth=2
		),
		yaxis=dict(
			title_text=f'Gold (GP) [{feature}]',
			titlefont=dict(
				family='Rockwell',
				size=12,
				color='white',
			),
			showline=True,
			showgrid=True,
			showticklabels=True,
			linecolor='white',
			linewidth=2,
			ticks='outside',
			tickfont=dict(
				family='Rockwell',
				size=12,
				color='white',
			),
		),
		showlegend=True,
		template = 'plotly_dark'

	)
	annotations = []
	annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
								xanchor='left', yanchor='bottom',
								text=title,
								font=dict(family='Rockwell',
											size=16,
											color='white'),
								showarrow=False))
	fig.update_layout(annotations=annotations)
	return fig
	
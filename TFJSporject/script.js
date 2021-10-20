console.log("Hello world from TFJS");

/**
* get the dataset, explore the data - visualize the data
*/

async function getdata(){
	const carsdata = await fetch("https://gist.githubusercontent.com/manashmandal/36a81de7f615855450a9c0265dd40c8c/raw/721d036d0fe0fc9b4adfad4413eef7520ef5e075/carsData.json");
	const cdata = await carsdata.json();
	const mydata = cdata.map(car =>({
		mpg:car.Miles_per_Gallon,
		hp:car.Horsepower,
	}))
	.filter(car=>(car.mpg!=null && car.hp!=null));
	return mydata;
}


function createmodel(){
	// create a deep learning model using sequential API
	const model = tf.sequential();
	model.add(tf.layers.dense({inputShape:[1],units:3,useBias:true}));
	// add the output Layer
	model.add(tf.layers.dense({units:1,useBias:true}));
	return model;
}


// Preprocessing of data - converting data into tensors, normalizing data 

function preprocess(data){
	return tf.tidy(() =>{
		// step 1 - shuffle the data
		tf.util.shuffle(data);
		
		// step 2 - converting data to tensors
		const x = data.map(d=>d.hp)
		const y = data.map(d=>d.mpg)
		
		const xtensor = tf.tensor2d(x, [x.length,1]);
		const ytensor = tf.tensor2d(y, [y.length,1]);
		
		// step3 - data normalizing using MinMaxscaling
		
		const xmin = xtensor.min();
		const xmax = xtensor.max();
		const ymin = ytensor.min();
		const ymax = ytensor.max();
		
		const xnorm = xtensor.sub(xmin).div(xmax.sub(xmin));
		const ynorm = ytensor.sub(ymin).div(ymax.sub(ymin));
		
		return {
			inputs:xnorm,
			outputs:ynorm,
			xmin,xmax,ymin,ymax,
		}
		});
}


async function train(model,x,y){
	model.compile({
		optimizer:tf.train.adam(),
		loss:tf.losses.meanSquaredError,
		metrics:['mse'],
	});
	const batchsize=64;
	const epochs=50;
	
	return await model.fit(x,y,
	{batchsize,
	epochs,
	callbacks:tfvis.show.fitCallbacks(
	{name:"Trainig performance"},
	['loss','mse'],
	{height:200,callbacks:['onEpochEnd']}
	)
	});
	}



async function main(){
	const data = await getdata();
	const values = data.map(d=>({
		x:d.hp,
		y:d.mpg,
	}));
	/*visualizing the data using tfvis - using scatter plot*/
	tfvis.render.scatterplot({name:"Horsepower v/s MPG"},
		{values},
			{xLabel:"Horsepower",
			yLabel:"MPG",
			height:300});
			
	const model = createmodel();
	tfvis.show.modelSummary({name:"Model Summary"},model);
	
	// get the data
	
	const tensordata = preprocess(data);
	const {inputs,outputs} = tensordata;
	
	// train the model
	
	await train(model,inputs,outputs);
	console.log("Done trainings");
}

document.addEventListener("DOMContentLoaded",main);
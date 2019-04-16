const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');
const nodewebcam = require('node-webcam');

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

/*
 *
 *	Importante fare il CROP della faccia
 *	perché ora fa il confronto dell'intera immagine
 *
 */

// via Promise
cv.imreadAsync('./faceimg.jpg')
	.then(img =>
		img
			.bgrToGrayAsync()
			.then(grayImg => classifier.detectMultiScaleAsync(grayImg))
			.then(res => {
				// @ts-ignore
				const { objects, numDetections } = res;
				// console.log('imreadAsync', res);
			})
	)
	.catch(err => console.error(err));

async function readfileAsync(path, arr) {
	return cv.imreadAsync(path).then(res => onReadFile(res, path, arr));
}

function onReadFile(img, path, arr) {
	return img.bgrToGrayAsync().then(grey => onImage(grey, path, arr));
}

function onImage(imgGrey, path, arr) {
	arr.push(imgGrey);
	console.log(path, 'processed');
}

let startTime, endTime;

function start() {
	startTime = new Date();
}

function end() {
	endTime = new Date();
	// @ts-ignore
	let timeDiff = endTime - startTime; //in ms
	// strip the ms
	timeDiff /= 1000;

	// get seconds
	const seconds = Math.round(timeDiff);
	return seconds;
}

async function initAsync() {
	// using async await
	try {
		start();
		console.log(`start processing images...`);

		const realtivepath = path.join(__dirname, '/images/trainers');
		const trainers = fs.readdirSync(realtivepath);
		const trainersPromiseArr = [];
		const trainersArr = [];

		// remove DS_STORE if exist
		if (trainers[0] == '.DS_Store') {
			trainers.splice(0, 1);
		}

		for (let i = 0; i < trainers.length; i++) {
			const imaegsFolder = path.join(realtivepath, trainers[i]);
			const images = fs.readdirSync(imaegsFolder);

			// remove DS_STORE if exist
			if (images[0] == '.DS_Store') {
				images.splice(0, 1);
			}

			for (let j = 0; j < images.length; j++) {
				trainersPromiseArr.push(
					readfileAsync(
						path.join(imaegsFolder, images[j]),
						trainersArr
					)
				);
			}
		}

		await Promise.all(trainersPromiseArr);

		const recognizer = new cv.LBPHFaceRecognizer();

		await recognizer.trainAsync(
			/*[grayImg]*/ trainersArr,
			trainersArr.map((v, i) => i)
		);

		let isRecognized = false;
		const cam = initCam();

		// do {
		cam.capture('captureToRecognize', async (err, data) => {
			console.log(data);
			// console.log(data);
			const img = await cv.imreadAsync(data);
			const grey = await img.bgrToGrayAsync();
			const face = await classifier.detectMultiScaleAsync(grey);

			console.log(face);

			// ciclo objects get rect find

			recognizer
				.predictAsync(grey)
				.then(res => {
					console.log('compare: ', res);
					if (res.confidence < 50) {
						console.log('hi giovanni');
						isRecognized = true;
					}
				})
				.catch(err => console.error(err));
		});
		// } while (!isRecognized);

		// @ts-ignore
		// const {
		// 	objects,
		// 	numDetections
		// } = await classifier.detectMultiScaleAsync(grayImg);

		console.log(`end recognize at ${end()} second`);
	} catch (err) {
		console.error(err);
	}
}

function initCam() {
	//Default options

	var opts = {
		//Picture related
		width: 1280,
		height: 720,
		quality: 100,
		//Delay to take shot
		delay: 0,
		//Save shots in memory
		saveShots: true,
		// [jpeg, png] support varies
		// Webcam.OutputTypes
		output: 'jpeg',
		//Which camera to use
		//Use Webcam.list() for results
		//false for default device
		device: false,
		// [location, buffer, base64]
		// Webcam.CallbackReturnTypes
		callbackReturn: 'location',
		//Logging
		verbose: false
	};

	return nodewebcam.create(opts);
}

initAsync();

// const isRecognized = false;
// const cam = initCam();

// do {
// 	cam.capture('captureToRecognize', (err, data) => {
// 		// console.log(data);
// 	});
// } while (isRecognized);

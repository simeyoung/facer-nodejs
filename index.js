// @ts-ignore
process.env.OPENCV4NODEJS_DISABLE_EXTERNAL_MEM_TRACKING = 1;

const fs = require('fs');
const path = require('path');
const cv = require('opencv4nodejs');

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const FPS = 10;
const wCap = new cv.VideoCapture(0);
wCap.set(cv.CAP_PROP_FRAME_WIDTH, 300);
wCap.set(cv.CAP_PROP_FRAME_HEIGHT, 300);

/*
 *
 *	Importante fare il CROP della faccia
 *	perché ora fa il confronto dell'intera immagine
 *
 */

async function readfileAsync(path, arr) {
	return cv.imreadAsync(path).then(res => onReadFile(res, path, arr));
}

function onReadFile(img, path, arr) {
	return img.bgrToGrayAsync().then(grey => onImage(grey, path, arr));
}

function onImage(imgGrey, path, arr) {
	const paths = path.split('/');
	const folderPerson = paths[paths.length - 2];
	arr.push({ grey: imgGrey, name: folderPerson });
	console.log(folderPerson, 'processed');
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
			trainersArr.map(v => v.grey),
			trainersArr.map((v, i) => i)
		);

		let isRecognized = false;

		setInterval(async () => {
			const frame = await wCap.readAsync();
			let grey = await frame.bgrToGrayAsync();
			const { objects } = await classifier.detectMultiScaleAsync(grey);

			if (!objects || objects.length == 0) {
				// console.error(`${path} non ho riconosciuto nessun viso`);
				return;
			}

			if (objects.length > 1) {
				// console.log(`ho trovate più di un viso`);
				return;
			}

			grey = grey.getRegion(objects[0]);

			recognizer
				.predictAsync(grey)
				.then(res => {
					const name = trainersArr
						.filter((x, i) => i == res.label)
						.map(v => v.name)[0];
					console.log('compare: ', { ...res, name });
					if (res.confidence < 50) {
						isRecognized = true;
						console.log(`hi ${name}`);
					}
				})
				.catch(err => console.error(err));
		}, 1000 / FPS);

		console.log(`end recognize at ${end()} second`);
	} catch (err) {
		console.error(err);
	}
}

initAsync();

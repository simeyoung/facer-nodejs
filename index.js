// @ts-ignore
process.env.OPENCV4NODEJS_DISABLE_EXTERNAL_MEM_TRACKING = 1;

const fs = require('fs');
const { LBPHFaceRecognizer, CascadeClassifier } = require('opencv4nodejs');
const cv = require('opencv4nodejs');
const path = require('path');
const express = require('express');
const app = express();
const server = require('http').createServer(app);
const io = require('socket.io')(server);
const { StreamCamera, Codec } = require('pi-camera-connect');

app.get('/', (req, res) => {
	res.sendFile(path.join(__dirname, 'index.html'));
});

server.listen(3000);

const classifier = new CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

const FPS = 10;
const TIMECOMPARE = 3;
const PORTNUMBER = 0;

let comparators = [];
let wCap;

// wCap = new cv.VideoCapture(PORTNUMBER);
// wCap.set(cv.CAP_PROP_FRAME_WIDTH, 300);
// wCap.set(cv.CAP_PROP_FRAME_HEIGHT, 300);

/**
 * @param {string} path
 * @param {any[]} arr
 */
async function readfileAsync(path, arr) {
	return cv.imreadAsync(path).then(res => onReadFile(res, path, arr));
}

/**
 * @param {any} img
 * @param {string} path
 * @param {{ grey: any; name: any; }[]} arr
 */
function onReadFile(img, path, arr) {
	/**
	 * @param {any} grey
	 */
	return img.bgrToGrayAsync().then(grey => onImage(grey, path, arr));
}

/**
 * @param {any} imgGrey
 * @param {string} path
 * @param {{ grey: any; name: any; }[]} arr
 */
function onImage(imgGrey, path, arr) {
	const paths = path.split('/');
	const folderPerson = paths[paths.length - 2];
	arr.push({ grey: imgGrey, name: folderPerson });
	console.log(folderPerson, 'processed');
}

/**
 * @param {{ label: any; confidence: any; }} res
 * @param {{grey: any; name: number}[]} trainers
 */
function onPrediction(res, trainers) {
	const name = trainers.filter((x, i) => i == res.label).map(v => v.name)[0];
	const prediction = { ...res, name };

	if (comparators.length == FPS * TIMECOMPARE) {
		// @ts-ignore
		const min = comparators.reduce(minExpression);
		// console.log(min);
		console.log(`hi ${min.name} confidence: ${min.confidence}`);
		comparators = [];

		io.emit('hello_man', min.name);
	}

	if (res.confidence < 100) {
		comparators.push(prediction);
	}
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

/**
 * @param {Buffer} img
 * @param {undefined} [rect]
 */
async function onEncodedAsync(img, rect) {
	const base64 = img.toString('base64');
	io.emit('image', base64);
}

/**
 * @param {{ predictAsync: (arg0: any) => { then: (arg0: (res: any) => void) => { catch: (arg0: (err: any) => void) => void; }; }; }} recognizer
 * @param {{ grey: any; name: number; }[]} trainersArr
 */
async function onFrame(recognizer, trainersArr, charData) {
	const rows = 100; // height
    const cols = 100; // width
    
    console.log(charData);
    return;

	const frame = new cv.Mat(charData, rows, cols);
	let grey = await frame.bgrToGrayAsync();
	const { objects } = await classifier.detectMultiScaleAsync(grey);

	let rect;
	let rectFounded = false;

	if (!objects || objects.length == 0) {
		// console.error(`${path} non ho riconosciuto nessun viso`);
		rect = null;
	} else if (objects.length > 1) {
		// console.log(`ho trovate più di un viso`);
		rect = null;
	} else {
		// viso corretto
		rect = objects[0];
	}

	if (rect) {
		grey = grey.getRegion(rect);
		frame.drawRectangle(rect, new cv.Vec3(0, 255, 0));
		rectFounded = true;
	}

	cv.imencodeAsync('.jpg', frame)
		.then(res => onEncodedAsync(res))
		.catch(err => console.log(err));

	if (!rectFounded) {
		return;
	}

	/**
	 * @param {{ label: any; confidence: any; }} res
	 */
	/**
	 * @param {any} err
	 */
	recognizer
		.predictAsync(grey)
		.then(res => onPrediction(res, trainersArr))
		.catch(err => console.error(err));
}

async function initAsync() {
	// using async await
	try {
		start();
		console.log(`start processing images...`);

		const realtivepath = path.join(__dirname, '/images/trainers');
		const trainers = fs.readdirSync(realtivepath);
		let trainersPromiseArr = [];
		let trainersArr = [];

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

		const recognizer = new LBPHFaceRecognizer();

		await recognizer.trainAsync(
			trainersArr.map(v => v.grey),
			trainersArr.map((v, i) => i)
		);

		let isRecognized = false;

		// setInterval(async () => {

		// }, 1000 / FPS);

		const stream = new StreamCamera({
			codec: Codec.H264,
			width: 640,
            height: 480,
            fps: 15
		});

		const video = stream.createStream();

		await stream.startCapture();

		// We can also listen to data events as they arrive
		video.on('data', async data => onFrame(recognizer, trainersArr, data));
		video.on('end', data => console.log('Video stream has ended'));

		console.log(`end recognize at ${end()} second`);
	} catch (err) {
		console.error(err);
	}
}

/**
 * @param {{ confidence: number; }} min
 * @param {{ confidence: number; }} next
 */
function minExpression(min, next) {
	return min.confidence < next.confidence ? min : next;
}

/*
 *   Start process
 */

initAsync();

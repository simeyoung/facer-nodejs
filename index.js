#!/usr/bin/env node
const cv = require('opencv4nodejs');
// @ts-ignore
// const picamera = require('raspberry-pi-camera-native');
const io = require('socket.io-client');
const socket = io('http://10.50.120.70:3000');

socket.on('connect', () => {
	console.log(`[SOCKET] websocket connected`);
	// socket.emit('server-connected', true);
});
socket.on('disconnect', () => console.log(`[SOCKET] websocket disconnected`));
socket.on('socket-started', () => console.log('socket started'));

socket.on('connection', sockett => {
	console.log(sockett);
});

// add frame data event listener
// picamera.on('frame', async frameData => {
// 	console.log(frameData);

// 	cv.imdecodeAsync(frameData)
// 		.then(() => console.log('eseguito sul cesso'))
// 		.catch(err => console.error(err));
// });

async function onFrame(charData) {
	try {
		// decode, transform gray, rescale and encode image
		const encoded = await cv
			.imdecodeAsync(charData)
			.then(frame => frame.bgrToGrayAsync())
			.then(grey => grey.rescaleAsync(0.5))
			.then(res => cv.imencodeAsync('.jpg', res));

		const sizeKB = encoded.byteLength / 1000 / 8;

		socket.emit('image', encoded.toString('base64'));
		console.log(`[SOCKET] sending buffer img ${sizeKB} KB`);
	} catch (err) {
		console.log(err);
	}
}

// picamera.on('frame', async charData => onFrame(charData));

const picameraOptions = {
	width: 640,
	height: 480,
	fps: 15,
	encoding: 'JPEG',
	quality: 50
};

// start capture
// picamera.start(picameraOptions);

const wCap = new cv.VideoCapture(0);
wCap.set(cv.CAP_PROP_FRAME_WIDTH, 300);
wCap.set(cv.CAP_PROP_FRAME_HEIGHT, 300);

setInterval(async () => {
	if (!socket.connected) {
		return;
	}

	const encoded = await wCap
		.readAsync()
		// .then(frame => frame.bgrToGrayAsync())
		.then(grey => grey.rescaleAsync(0.5))
		// .then(res => res.getDataAsync())
		.then(res => cv.imencodeAsync('.jpg', res));

	const sizeKB = encoded.byteLength / 1000 / 8;

	socket
		// verificare la corretta
		// compressione del buffer
		// .compress(false)
		.emit('imageToAnalyze', encoded /*.toString('base64')*/);
	console.log(`[SOCKET] sending buffer img ${sizeKB} KB`);
}, 1000 / 15);

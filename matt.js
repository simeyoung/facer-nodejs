const cv = require('opencv4nodejs');
const path = require('path');
const picamera = require('raspberry-pi-camera-native');

// const file = path.join(__dirname, 'video-stream.h264');
// cv.imreadAsync(file)
// 	.then(res => {
// 		const type = res.type;
// 		console.log('mat type: ' + type);
// 	})
// 	.catch(err => console.error(err));

// add frame data event listener
picamera.on('frame', async frameData => {
	console.log(frameData);

	cv.imdecodeAsync(frameData)
		.then(() => console.log('eseguito sul cesso'))
		.catch(err => console.error(err));
});

// start capture
picamera.start();

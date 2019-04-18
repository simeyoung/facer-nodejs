const cv = require('opencv4nodejs');
const path = require('path');

const file = path.join(__dirname, 'video-stream.h264');
cv.imreadAsync(file)
	.then(res => {
		const type = res.type;
		console.log('mat type: ' + type);
	})
	.catch(err => console.error(err));

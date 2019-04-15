const cv = require('opencv4nodejs');
// import directly
const FaceRecognizer = cv.FaceRecognizer;
const Mat = cv.Mat;

const classifier = new cv.CascadeClassifier(cv.HAAR_FRONTALFACE_ALT2);

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

async function initAsync() {
	// using async await
	try {
		console.log('start recognize..');
		const img = await cv.imreadAsync('./faceimg.jpg');
		const grayImg = await img.bgrToGrayAsync();

		const imgCompare1 = await cv.imreadAsync('./facecompare1.jpg');
		const imgCompare2 = await cv.imreadAsync('./facecompare2.jpg');

		const imgCompare1Gray = await imgCompare1.bgrToGrayAsync();
		const imgCompare2Gray = await imgCompare2.bgrToGrayAsync();

		const recognizer = new cv.LBPHFaceRecognizer();

		await recognizer.trainAsync([grayImg], [1]);

		// compare 1
		recognizer
			.predictAsync(imgCompare1Gray)
			.then(res => console.log('compare 1: ', res))
			.catch(err => console.error(err));
		// compare 2
		recognizer
			.predictAsync(imgCompare2Gray)
			.then(res => console.log('compare 2: ', res))
			.catch(err => console.error(err));

		// @ts-ignore
		// const {
		// 	objects,
		// 	numDetections
		// } = await classifier.detectMultiScaleAsync(grayImg);

		console.log('end recognize..');
	} catch (err) {
		console.error(err);
	}
}

initAsync();

const containerVideo = document.querySelector('#containerVideo');
const containerInfo = document.querySelector('#containerInfo');

async function loadTrainingData(){
    const labels = ['Ngoc An', 'Hong Dao'];
    const faceDescriptors = [];
    for (const label of labels) {
        const descriptors = []
        for (let i = 1; i <= 8; i++) {
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpeg`);
            const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
            console.log(i);
            console.log(detection);
            descriptors.push(detection.descriptor);
        }
        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors))
        Toastify({text: `Training Done of ${label}`}).showToast();
    }
    return faceDescriptors;
}

// let faceMatcher
let faceMatcherVideo;
let currentImage = null;
let peopleName = null;
let peopleTempName;
let input;
async function init(){
    await Promise.all([
        await faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        await faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        await faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
        await faceapi.nets.tinyFaceDetector.loadFromUri('/models'),
        await faceapi.nets.faceExpressionNet.loadFromUri('/models'),
        await faceapi.nets.ageGenderNet.loadFromUri('/models'),
    ])

    const trainingData = await loadTrainingData();
    faceMatcherVideo = new faceapi.FaceMatcher(trainingData, 0.6);

    const video = await startVideo();   
    const canvasVideo = faceapi.createCanvasFromMedia(video);
    containerVideo.appendChild(video);  
    containerVideo.appendChild(canvasVideo);

    const sizeVideo = {width: video.videoWidth, height: video.videoHeight};
    faceapi.matchDimensions(canvasVideo, sizeVideo);

    setInterval(async () => {
        const detectionsVideo = await faceapi.detectAllFaces(video).withFaceLandmarks().withFaceExpressions().withAgeAndGender().withFaceDescriptors()
        const resizedDetectionsVideo = faceapi.resizeResults(detectionsVideo, sizeVideo);
        canvasVideo.getContext('2d').clearRect(0, 0, video.videoWidth, video.videoHeight);

        faceapi.draw.drawDetections(canvasVideo, resizedDetectionsVideo);
        faceapi.draw.drawFaceLandmarks(canvasVideo, resizedDetectionsVideo);
        faceapi.draw.drawFaceExpressions(canvasVideo, resizedDetectionsVideo);

        for (const detection of resizedDetectionsVideo) {
            const box = detection.detection.box;
            let name = faceMatcherVideo.findBestMatch(detection.descriptor).toString().replace(/\s*\([^)]*\)/, '');
            let age = Math.round(detection.age);
            let gender = detection.gender;
            if (gender === 'male') {gender = 'nam'}
            else {gender = 'nữ';}
            let score = detection.detection._score.toFixed(2);

            const drawBox = new faceapi.draw.DrawBox(box, {
                label: name + " (" + score + ") " +  age + " " + gender})
            drawBox.draw(canvasVideo);
        }
    }, 100);
    
    Toastify({text: "Success Upload Model"}).showToast();
}

init()

async function startVideo() {
    const video = document.createElement('video');
    video.srcObject = await navigator.mediaDevices.getUserMedia({ video: true });
    await video.play();
    return video;
}
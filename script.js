const imageUpload = document.getElementById("imageUpload");
let labelPercent = 0;
let labelCount = 0;
const loadingDiv = document.getElementById("loading_state");

loadingDiv.innerHTML = `Loading models`;
document.getElementById("loading").style.display = "block";

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models")
]).then(start);

document.getElementById("imageUpload").style.display = "none";

async function start() {
  loadingDiv.innerHTML = "Done loading models";
  const container = document.createElement("div");
  container.style.position = "relative";
  document.body.append(container);
  const labeledFaceDescriptors = await loadLabeledImages();
  loadingDiv.innerHTML = `Teaching model images`;

  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.5);
  let image;
  let canvas;
  document.getElementById("imageUpload").style.display = "block";
  document.getElementById("loading").style.display = "none";
  imageUpload.addEventListener("change", async () => {
    if (image) image.remove();
    if (canvas) canvas.remove();
    image = await faceapi.bufferToImage(imageUpload.files[0]);
    container.append(image);
    canvas = faceapi.createCanvasFromMedia(image);
    container.append(canvas);
    const displaySize = { width: image.width, height: image.height };
    faceapi.matchDimensions(canvas, displaySize);
    const detections = await faceapi
      .detectAllFaces(image)
      .withFaceLandmarks()
      .withFaceDescriptors();
    const resizedDetections = faceapi.resizeResults(detections, displaySize);
    const results = resizedDetections.map(d =>
      faceMatcher.findBestMatch(d.descriptor)
    );
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box;
      const drawBox = new faceapi.draw.DrawBox(box, {
        label: result.toString()
      });
      drawBox.draw(canvas);
    });
  });
}

function loadLabeledImages() {
  const labels = [
    "Black Widow",
    "Captain America",
    "Captain Marvel",
    "Hawkeye",
    "Jim Rhodes",
    "Thor",
    "Tony Stark",
    "Black Panther",
    "Spider Man"
  ];
  return Promise.all(
    labels.map(async label => {
      labelCount++;
      labelPercent = labelCount / (labels.length) * 100;
      console.log(`${Math.round(labelPercent)}%`);
      loadingDiv.innerHTML = `Downloading model images: ${Math.round(labelPercent)}%`;
      if(labelPercent == 100) loadingDiv.style.display = "none";
      const descriptions = [];
      for (let i = 1; i <= 2; i++) {
        const img = await faceapi.fetchImage(
          `./labeled_images/${label}/${i}.jpg`
        );
        const detections = await faceapi
          .detectSingleFace(img)
          .withFaceLandmarks()
          .withFaceDescriptor();
        descriptions.push(detections.descriptor);
      }

      return new faceapi.LabeledFaceDescriptors(label, descriptions);
    })
    
  );
  
}

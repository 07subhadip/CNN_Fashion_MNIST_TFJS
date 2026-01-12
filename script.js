const CLASS_NAMES = [
    "T-shirt/top 👕",
    "Trouser 👖",
    "Pullover 🧥",
    "Dress 👗",
    "Coat 🧥",
    "Sandal 👡",
    "Shirt 👔",
    "Sneaker 👟",
    "Bag 👜",
    "Ankle boot 🥾",
];

const CANVAS_SIZE = 400;
const MODEL_INPUT_SIZE = 28;

let isDrawing = false;
let models = { cnn: null, fnn: null };
let currentModelName = "cnn";

const canvas = document.getElementById("blackboard");
const ctx = canvas.getContext("2d", { willReadFrequently: true });
const resultText = document.getElementById("prediction-result");
const confidenceScore = document.getElementById("confidence-score");
const confidenceFill = document.getElementById("confidence-fill");
const modelStatus = document.getElementById("model-status");

async function init() {
    setupCanvas();
    await tf.ready();
    const backend = tf.getBackend();
    console.log(`TF.js initialized. Backend: ${backend}`);
    loadModel("cnn");
}

function setupCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    ctx.strokeStyle = "white";
    ctx.lineWidth = 15;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";

    canvas.addEventListener("mousedown", startDrawing);
    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", stopDrawing);
    canvas.addEventListener("mouseout", stopDrawing);

    canvas.addEventListener(
        "touchstart",
        (e) => {
            if (e.target === canvas) e.preventDefault();
            startDrawing(e);
        },
        { passive: false }
    );

    canvas.addEventListener(
        "touchmove",
        (e) => {
            if (e.target === canvas) e.preventDefault();
            draw(e);
        },
        { passive: false }
    );

    canvas.addEventListener("touchend", stopDrawing);
}

function getPointerPos(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;

    return {
        x: clientX - rect.left,
        y: clientY - rect.top,
    };
}

function startDrawing(e) {
    isDrawing = true;
    const { x, y } = getPointerPos(e);
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function draw(e) {
    if (!isDrawing) return;
    const { x, y } = getPointerPos(e);
    ctx.lineTo(x, y);
    ctx.stroke();
}

function stopDrawing() {
    if (isDrawing) {
        isDrawing = false;
        ctx.closePath();
    }
}

document.getElementById("model-select").addEventListener("change", (e) => {
    currentModelName = e.target.value;
    loadModel(currentModelName);
});

async function loadModel(name) {
    if (models[name]) {
        updateStatus(name, "Ready", "#00f2ff");
        currentModelName = name;
        return;
    }

    const modelUrl = `./web_model/${name}/model.json`;
    updateStatus(name, "Loading...", "#ffcc00");

    try {
        models[name] = await loadModelWithPatch(modelUrl);

        tf.tidy(() => {
            const dummy = tf.zeros([1, 28, 28, 1]);
            try {
                models[name].predict(dummy);
            } catch (e) {
                models[name].predict(dummy.flatten().expandDims(0));
            }
        });

        updateStatus(name, "Ready", "#00f2ff");
    } catch (err) {
        console.error("Critical Error Loading Model:", err);
        updateStatus(name, "Error", "#ff0055");
        alert(`Failed to load ${name} model. Check console for details.`);
    }
}

function updateStatus(name, status, color) {
    modelStatus.innerText = `${name.toUpperCase()} ${status}`;
    modelStatus.style.color = color;
}

async function loadModelWithPatch(url) {
    const response = await fetch(url);
    const modelJson = await response.json();

    const config = modelJson.modelTopology?.model_config?.config;
    if (config?.layers) {
        config.layers.forEach((layer) => {
            if (layer.class_name === "InputLayer" && layer.config) {
                if (layer.config.batch_shape && !layer.config.batchInputShape) {
                    layer.config.batchInputShape = layer.config.batch_shape;
                }
            }
        });
    }

    const weightPathPrefix = url.substring(0, url.lastIndexOf("/") + 1);
    const customHandler = {
        load: async () => {
            const weightsManifest = modelJson.weightsManifest;
            const weightFiles = weightsManifest[0].paths;

            const weightBuffers = await Promise.all(
                weightFiles.map(async (file) => {
                    const res = await fetch(weightPathPrefix + file);
                    return await res.arrayBuffer();
                })
            );

            const totalLength = weightBuffers.reduce(
                (acc, buf) => acc + buf.byteLength,
                0
            );
            const combinedBuffer = new Uint8Array(totalLength);
            let offset = 0;
            for (const buf of weightBuffers) {
                combinedBuffer.set(new Uint8Array(buf), offset);
                offset += buf.byteLength;
            }

            return {
                modelTopology: modelJson.modelTopology,
                weightSpecs: weightsManifest[0].weights,
                weightData: combinedBuffer.buffer,
            };
        },
    };

    return await tf.loadLayersModel(customHandler);
}

document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("clear-btn").addEventListener("click", clearBoard);

function clearBoard() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    resultText.innerText = "---";
    confidenceScore.innerText = "0%";
    confidenceFill.style.width = "0%";
}

async function predict() {
    const model = models[currentModelName];
    if (!model) {
        alert("Model not loaded yet.");
        return;
    }

    const data = tf.tidy(() => {
        const canvasTensor = tf.browser.fromPixels(canvas);

        const resized = tf.image.resizeBilinear(canvasTensor, [
            MODEL_INPUT_SIZE,
            MODEL_INPUT_SIZE,
        ]);

        const grayscale = resized.mean(2);

        let input = grayscale.expandDims(0).expandDims(-1);

        input = input.div(255.0);

        const inputShape = model.inputs[0].shape;
        if (inputShape.length === 2 && inputShape[1] === 784) {
            input = input.flatten().expandDims(0);
        }

        const prediction = model.predict(input);
        return prediction.dataSync();
    });

    const maxProb = Math.max(...data);
    const predIndex = data.indexOf(maxProb);

    resultText.innerText = CLASS_NAMES[predIndex];
    const confPercent = (maxProb * 100).toFixed(1);
    confidenceScore.innerText = `${confPercent}%`;
    confidenceFill.style.width = `${confPercent}%`;

    console.log(`Prediction: ${CLASS_NAMES[predIndex]} (${confPercent}%)`);
}

init();

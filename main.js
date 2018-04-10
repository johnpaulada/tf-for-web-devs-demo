let mobnet = {},
    voteModel = {}

init()

function init() {
  document.querySelector('#train').addEventListener('click', () => {
    document.querySelector('.notification').style.display = "block";
    train()
  })
  document.querySelector('#predict').addEventListener('click', () => {
    predict()
  })
}

async function train() {
  setStatus("Training...")
  mobnet = await getMobnet()

  const trainingData = getData()

  voteModel = getModel()

  await voteModel.fit(trainingData.x, trainingData.y, {
    batchSize: 2,
    epochs: 200
  })

  setStatus("Training Complete!")
}

async function predict() {
  // Set image preview
  const urlInput = document.querySelector('#fileUrl')
  const image = document.querySelector('#preview')
  image.src = `./public/ballots/${urlInput.value}.png`

  const mobnetInput = imageToInput(image)
  const mobnetOutput = mobnet.predict(mobnetInput)
  const voteOutput = voteModel.predict(mobnetOutput)
  
  alert((await voteOutput.data()).map(v => v > 0.9).reduce((p, v, i) => {
    return v ? `${p} ${["Duterte", "Miriam", "Binay"][i]}` : p
  }, "Voted for:"))
}

function getModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: 48,
        activation: 'relu',
      }),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({
        units: 48,
        activation: 'relu',
      }),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
      })
    ]
  })

  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'})

  return model
}

async function getMobnet() {
  const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  const mobileNet = await tf.loadModel(MOBILENET_URL)
  const mobnetOutput = mobileNet.getLayer('conv_pw_13_relu')
  const mobnetModel = tf.model({inputs: mobileNet.inputs, outputs: mobnetOutput.output})

  return mobnetModel
}

function getData() {
  return tf.tidy(() => {
    const trainingData = {x: [], y: []}
    const rawData = getRawData()
  
    trainingData.y = tf.tensor(rawData.y)
    trainingData.x = rawData.x.map(rawXToImage)
    trainingData.x = trainingData.x.map(imageToInput)
    trainingData.x = trainingData.x.map(inputToPredictedInput)
    trainingData.x = trainingData.x.reduce((p, c) => p.concat(c))
  
    return trainingData
  })
}

function rawXToImage(rawX) {
    const image = document.createElement('img');
    image.src = `./public/ballots/${rawX}.png`; 

    return image
}

function imageToInput(image) {
  return tf.tidy(() => {
    const tfImage = tf.fromPixels(image)
    const resizedImage = tf.image.resizeBilinear(tfImage, [224, 224])
    const batchedImage = resizedImage.expandDims(0)
    const normalizedImage = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))
  
    return normalizedImage
  })
}

function inputToPredictedInput(input) {
  return mobnet.predict(input)
}

function getRawData() {
  return {
    x: ['Ballot-000-1', 'Ballot-000-2', 'Ballot-000-4',
      'Ballot-000-5', 'Ballot-000-6', 'Ballot-000-7',
      'Ballot-000-8', 'Ballot-000-24',

      'Ballot-001-11', 'Ballot-001-13', 'Ballot-001-16', 'Ballot-001-22', 'Ballot-001-23',

      'Ballot-010-18',
    
      'Ballot-011-20', 'Ballot-011-21',

      'Ballot-100-9', 'Ballot-100-25',

      'Ballot-101-17',
    
      'Ballot-110-10', 'Ballot-110-15',
    
      'Ballot-111-2', 'Ballot-111-14', 'Ballot-111-19'],
    y: [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],

      [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1],

      [0, 1, 0],

      [0, 1, 1], [0, 1, 0],

      [1, 0 ,0], [1, 0 ,0],

      [1, 0, 1],
      
      [1, 1, 0], [1, 1, 0],
    
      [1, 1, 1], [1, 1, 1], [1, 1, 1]]
  }
}

async function setStatus(status) {
  document.querySelector('#status').innerHTML = `${status}`
}
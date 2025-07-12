import { BedrockImageManipulation, type TaskType } from '../src/index';

import fs from 'node:fs';
import path from 'node:path';

const readImageAsBase64 = (filename: string): string => {
  const imagePath = path.join(__dirname, '..', 'images', filename);
  const imageBuffer = fs.readFileSync(imagePath);
  return imageBuffer.toString('base64');
};

const writeImageFromBase64 = (base64Data: string, outputFilename: string) => {
  const outputPath = path.join(__dirname, '..', 'results', outputFilename);
  const buffer = Buffer.from(base64Data, 'base64');
  fs.writeFileSync(outputPath, buffer);
};

const run = async () => {
  const imageManipulator = new BedrockImageManipulation();

  const tasks: {
    taskType: TaskType;
    inputFiles: string[];
    outputFile: string;
    prompt?: string;
    configOverrides?: Partial<
      Parameters<typeof imageManipulator.generateImage>[0]
    >;
  }[] = [
    {
      taskType: 'VIRTUAL_TRY_ON',
      inputFiles: [], // not required for VIRTUAL_TRY_ON
      outputFile: 'virtual-try-on-1.png',
      configOverrides: {
        virtualTryOnParams: {
          sourceImage: readImageAsBase64('source-model-1.png'),
          referenceImage: readImageAsBase64('reference-garment-1.png'),
          maskType: 'GARMENT',
          garmentBasedMask: {
            maskShape: 'CONTOUR',
            garmentClass: 'UPPER_BODY',
            garmentStyling: {
              longSleeveStyle: 'SLEEVE_UP',
              tuckingStyle: 'TUCKED',
              outerLayerStyle: 'OPEN',
            },
          },
          maskExclusions: {
            preserveFace: 'ON',
            preserveHands: 'ON',
            preserveBodyPose: 'ON',
          },
          mergeStyle: 'SEAMLESS',
          returnMask: false,
        },
        imageConfig: {
          numberOfImages: 1,
          quality: 'premium',
          cfgScale: 7.5,
        },
      },
    },
    {
      taskType: 'VIRTUAL_TRY_ON',
      inputFiles: [], // not required for VIRTUAL_TRY_ON
      outputFile: 'virtual-try-on-2.png',
      configOverrides: {
        virtualTryOnParams: {
          sourceImage: readImageAsBase64('source-model-2.png'),
          referenceImage: readImageAsBase64('reference-garment-1.png'),
          maskType: 'GARMENT',
          garmentBasedMask: {
            maskShape: 'CONTOUR',
            garmentClass: 'UPPER_BODY',
            garmentStyling: {
              longSleeveStyle: 'SLEEVE_UP',
              tuckingStyle: 'TUCKED',
              outerLayerStyle: 'OPEN',
            },
          },
          maskExclusions: {
            preserveFace: 'ON',
            preserveHands: 'ON',
            preserveBodyPose: 'ON',
          },
          mergeStyle: 'SEAMLESS',
          returnMask: false,
        },
        imageConfig: {
          numberOfImages: 1,
          quality: 'premium',
          cfgScale: 7.5,
        },
      },
    },
    {
      taskType: 'VIRTUAL_TRY_ON',
      inputFiles: [], // not required for VIRTUAL_TRY_ON
      outputFile: 'virtual-try-on-3.png',
      configOverrides: {
        virtualTryOnParams: {
          sourceImage: readImageAsBase64('source-room-1.png'),
          referenceImage: readImageAsBase64('reference-sofa-1.png'),
          maskType: 'PROMPT',
          promptBasedMask: {
            maskShape: 'BOUNDING_BOX',
            maskPrompt: 'Replace the existing sofa',
          },
          mergeStyle: 'SEAMLESS',
          returnMask: false,
        },
        imageConfig: {
          numberOfImages: 1,
          seed: 23, // random seed
          quality: 'premium',
          cfgScale: 7.5,
        },
      },
    },
    {
      taskType: 'VIRTUAL_TRY_ON',
      inputFiles: [], // not required for VIRTUAL_TRY_ON
      outputFile: 'virtual-try-on-4.png',
      configOverrides: {
        virtualTryOnParams: {
          sourceImage: readImageAsBase64('source-room-2.png'),
          referenceImage: readImageAsBase64('reference-sofa-1.png'),
          maskType: 'PROMPT',
          promptBasedMask: {
            maskShape: 'BOUNDING_BOX',
            maskPrompt: 'Replace the existing sofa',
          },
          mergeStyle: 'SEAMLESS',
          returnMask: false,
        },
        imageConfig: {
          numberOfImages: 1,
          seed: 23, // random seed
          quality: 'premium',
          cfgScale: 7.5,
        },
      },
    },
    {
      taskType: 'VIRTUAL_TRY_ON',
      inputFiles: [], // not required for VIRTUAL_TRY_ON
      outputFile: 'virtual-try-on-5.png',
      configOverrides: {
        virtualTryOnParams: {
          sourceImage: readImageAsBase64('source-room-4.jpg'),
          referenceImage: readImageAsBase64('reference-sofa-2.png'),
          maskType: 'PROMPT',
          promptBasedMask: {
            maskShape: 'BOUNDING_BOX',
            maskPrompt: 'Replace only the sofa',
          },
          mergeStyle: 'DETAILED',
          returnMask: false,
        },
        imageConfig: {
          numberOfImages: 1,
          quality: 'premium',
          cfgScale: 7.5,
        },
      },
    },
  ];

  for (const [index, task] of tasks.entries()) {
    console.log(`Running task ${index + 1}: ${task.taskType}`);

    const inputImage = task.inputFiles[0]
      ? readImageAsBase64(task.inputFiles[0])
      : undefined;

    const maskImage =
      (task.taskType === 'INPAINTING' || task.taskType === 'OUTPAINTING') &&
      !task.configOverrides?.maskPrompt
        ? readImageAsBase64(task.inputFiles[1])
        : undefined;

    const generationResult = await imageManipulator.generateImage({
      modelId: 'amazon.nova-canvas-v1:0',
      taskType: task.taskType,
      prompt: task.prompt,
      base64Image: inputImage,
      maskImage: maskImage,
      imageConfig: {
        width: 512,
        height: 512,
        quality: 'premium',
      },
      ...task.configOverrides,
    });

    writeImageFromBase64(generationResult.base64Image, task.outputFile);
  }

  console.log('All tasks complete. Results saved to /results folder.');
};

run().catch((error) => {
  console.error('Failed to generate image:', error);
});

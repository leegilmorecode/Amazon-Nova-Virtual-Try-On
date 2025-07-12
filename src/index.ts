import {
  BedrockRuntimeClient,
  InvokeModelCommand,
} from '@aws-sdk/client-bedrock-runtime';

/**
 * Configuration object for image generation
 */
export type TaskType =
  | 'TEXT_IMAGE'
  | 'INPAINTING'
  | 'OUTPAINTING'
  | 'BACKGROUND_REMOVAL'
  | 'COLOR_GUIDED_GENERATION'
  | 'VIRTUAL_TRY_ON';

export interface ImageGenerationInput {
  prompt?: string;
  modelId?: string;
  region?: string;
  taskType: TaskType;
  virtualTryOnParams?: {
    sourceImage: string;
    referenceImage: string;
    maskType: 'IMAGE' | 'GARMENT' | 'PROMPT';
    imageBasedMask?: {
      maskImage: string;
    };
    garmentBasedMask?: {
      maskShape: 'CONTOUR' | 'BOUNDING_BOX' | 'DEFAULT';
      garmentClass:
        | 'UPPER_BODY'
        | 'LOWER_BODY'
        | 'FULL_BODY'
        | 'FOOTWEAR'
        | 'LONG_SLEEVE_SHIRT'
        | 'SHORT_SLEEVE_SHIRT'
        | 'NO_SLEEVE_SHIRT'
        | 'OTHER_UPPER_BODY'
        | 'LONG_PANTS'
        | 'SHORT_PANTS'
        | 'OTHER_LOWER_BODY'
        | 'LONG_DRESS'
        | 'SHORT_DRESS'
        | 'FULL_BODY_OUTFIT'
        | 'OTHER_FULL_BODY'
        | 'SHOES'
        | 'BOOTS'
        | 'OTHER_FOOTWEAR';
      garmentStyling?: {
        longSleeveStyle?: 'SLEEVE_DOWN' | 'SLEEVE_UP';
        tuckingStyle?: 'UNTUCKED' | 'TUCKED';
        outerLayerStyle?: 'CLOSED' | 'OPEN';
      };
    };
    promptBasedMask?: {
      maskShape: 'BOUNDING_BOX' | 'CONTOUR' | 'DEFAULT';
      maskPrompt: string;
    };
    maskExclusions?: {
      preserveBodyPose?: 'ON' | 'OFF' | 'DEFAULT';
      preserveHands?: 'ON' | 'OFF' | 'DEFAULT';
      preserveFace?: 'ON' | 'OFF' | 'DEFAULT';
    };
    mergeStyle?: 'BALANCED' | 'SEAMLESS' | 'DETAILED';
    returnMask?: boolean;
  };

  imageConfig?: {
    width?: number;
    height?: number;
    quality?: 'standard' | 'premium';
    cfgScale?: number;
    seed?: number;
    numberOfImages?: number;
  };

  // Common image inputs
  base64Image?: string;
  maskImage?: string;
  maskPrompt?: string;
  negativePrompt?: string;
  outPaintingMode?: 'DEFAULT' | 'PRECISE';
  conditionImage?: string;
  controlMode?: 'CANNY_EDGE' | 'SEGMENTATION';
  controlStrength?: number;
  colorHexList?: string[]; // e.g. ['#FF0000']
}

/**
 * Result of image generation
 */
export interface ImageGenerationResult {
  /** Raw image bytes */
  imageBytes: Uint8Array;
  /** Base64-encoded image string */
  base64Image: string;
  /** Model ID used for generation */
  modelId: string;
}

/**
 * Main class for generating and manipulating images
 */
export class BedrockImageManipulation {
  /** AWS Bedrock client instance */
  private bedrock: BedrockRuntimeClient;
  /** AWS region */
  private region: string;

  constructor(options: { region?: string } = {}) {
    this.region = options.region || 'us-east-1';
    this.bedrock = new BedrockRuntimeClient({ region: this.region });
  }

  /**
   * Generates image bytes from a text prompt using Amazon Bedrock's image generation models.
   */
  public async generateImage(
    options: ImageGenerationInput,
  ): Promise<ImageGenerationResult> {
    const {
      taskType,
      prompt,
      modelId = 'amazon.nova-canvas-v1:0',
      region = this.region,
      imageConfig = {},
      base64Image,
      maskImage,
      maskPrompt,
      negativePrompt,
      outPaintingMode,
      conditionImage,
      controlMode,
      controlStrength,
      colorHexList,
    } = options;

    const {
      width = 1024,
      height = 1024,
      quality = 'premium',
      cfgScale = 8.0,
      seed,
      numberOfImages = 1,
    } = imageConfig;

    const client =
      region === this.region
        ? this.bedrock
        : new BedrockRuntimeClient({ region });

    // Construct dynamic request body based on task type
    const requestBody: Record<
      string,
      string | number | boolean | object | null
    > = {
      taskType,
      imageGenerationConfig: {
        numberOfImages,
        quality,
        cfgScale,
        ...(seed !== undefined && { seed }),
        ...(width && { width }),
        ...(height && { height }),
      },
    };

    switch (taskType) {
      case 'TEXT_IMAGE':
        requestBody.textToImageParams = {
          text: prompt,
          negativeText: negativePrompt,
          ...(conditionImage && {
            conditionImage,
            controlMode,
            controlStrength,
          }),
        };
        break;

      case 'INPAINTING':
        requestBody.inPaintingParams = {
          image: base64Image,
          maskImage,
          maskPrompt,
          text: prompt,
          negativeText: negativePrompt,
        };
        break;

      case 'OUTPAINTING':
        requestBody.outPaintingParams = {
          image: base64Image,
          maskImage,
          maskPrompt,
          text: prompt,
          negativeText: negativePrompt,
          outPaintingMode: outPaintingMode || 'DEFAULT',
        };
        break;

      case 'BACKGROUND_REMOVAL':
        requestBody.backgroundRemovalParams = {
          image: base64Image,
        };
        break;

      case 'COLOR_GUIDED_GENERATION':
        requestBody.colorGuidedGenerationParams = {
          text: prompt,
          negativeText: negativePrompt,
          referenceImage: base64Image,
          colors: colorHexList,
        };
        break;

      case 'VIRTUAL_TRY_ON':
        if (!options.virtualTryOnParams) {
          throw new Error(
            'virtualTryOnParams must be provided for VIRTUAL_TRY_ON',
          );
        }

        requestBody.virtualTryOnParams = options.virtualTryOnParams;
        break;
      default:
        throw new Error(`Unsupported taskType: ${taskType}`);
    }

    const command = new InvokeModelCommand({
      modelId,
      body: JSON.stringify(requestBody),
      contentType: 'application/json',
      accept: 'application/json',
    });

    try {
      const response = await client.send(command);

      if (!response.body) {
        throw new Error('No response body received from model');
      }

      const responseBody = JSON.parse(new TextDecoder().decode(response.body));

      if (responseBody.error) {
        throw new Error(`Model error: ${responseBody.error}`);
      }

      if (!responseBody.images || !responseBody.images[0]) {
        throw new Error('No image returned from model');
      }

      const base64Image = responseBody.images[0];
      const imageBytes = new Uint8Array(Buffer.from(base64Image, 'base64'));

      return {
        imageBytes,
        base64Image,
        modelId,
      };
    } catch (error) {
      throw new Error(
        `Image generation failed: ${
          error instanceof Error ? error.message : String(error)
        }`,
      );
    }
  }
}

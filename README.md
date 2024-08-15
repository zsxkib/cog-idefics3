# Idefics3

<a href='https://replicate.com/zsxkib/idefics3'><img src='https://replicate.com/zsxkib/idefics3/badge'></a>

This is a Cog model of Idefics3-8B-Llama3, a smart AI that works with both images and text.

## How to Use

To use this model, run a command like this:

```bash
sudo cog predict \
  -i image="https://images.dog.ceo/breeds/terrier-dandie/n02096437_3848.jpg" \
  -i text="What do we see in this image?" \
  -i assistant_prefix="Let's think step by step." \
  -i decoding_strategy="top-p-sampling" \
  -i temperature=0.4 \
  -i max_new_tokens=512 \
  -i repetition_penalty=1.2 \
  -i top_p=0.8
...
Running prediction...
The image depicts a small dog with long, shaggy fur. This breed appears to be a Yorkshire Terrier (Yorkie), known for its distinctive coat and size. The Yorkie has light brown and black hair that is well-groomed but slightly tousled, giving it an endearing appearance.

The dog's eyes are dark and expressive, looking directly at the camera. Its ears stand upright, adding to its alert demeanor. The nose of the dog is also visible, which complements its overall facial features.

The background shows what seems like a tiled floor, possibly made from ceramic or stone tiles arranged in a pattern resembling a grid. There might be some indistinct objects on the edges of the frame, suggesting there could be furniture or other items nearby, though they aren't clearly identifiable due to their blurred nature.

### Analysis:

**Breed Identification:**
- **Name**: Yorkshire Terrier (commonly referred to as "Yorkies")
- **Physical Characteristics**: Long-haired, shaggy fur; typically weighs between 7–15 pounds (3–6 kg) and stands about 10 inches tall.

**Behavioral Observations**:
- Dogs often have unique personalities ranging from playful to affectionate, making them popular pets worldwide.

**Environmental Context**:
- Tiled floors can indicate either indoor or outdoor settings depending on weather conditions. They provide durability and ease of cleaning compared to carpets.

Given these observations, if one were to ask questions related to the image such as identifying the breed, understanding the environment, or speculating on potential actions of the animal based on visual cues, here’s how you would approach answering those queries using Chain of Thought reasoning:

1. **Identification by Breed**:
   - Based on the description provided, the dog resembles a Yorkie given its characteristic long, shaggy fur and medium-sized body structure.

2. **Environment Speculation**:
   - The presence of a tiled floor suggests an area designed for easy maintenance, potentially indoors where cleanliness is important. However, without additional context, whether it’s inside or outside cannot be definitively determined.

3. **Potential Actions/Behaviors**:
   - Given the direct gaze of the dog, it may be curious or attentive towards something off-camera, perhaps another pet or person not shown in the picture.

This detailed analysis provides comprehensive insights into both the physical characteristics and contextual elements present within the image. By integrating relevant knowledge about breeds and environmental contexts, any further inquiries regarding the depicted scene can be logically addressed.
```

## What It Does

This model can:
- Describe images in detail
- Answer questions about pictures
- Analyze both images and text together

## Input Options

You can set these options when using the model:

- `image`: Your image file (required)
- `text`: Your question or comment about the image (required)
- `assistant_prefix`: How you want the AI to act (default: "Let's think step by step.")
- `decoding_strategy`: How to create the answer ("greedy" or "top-p-sampling", default: "greedy")
- `temperature`: How creative the AI should be (0.0 to 5.0, default: 0.4)
- `max_new_tokens`: How long the answer can be (8 to 1024, default: 512)
- `repetition_penalty`: Helps avoid repeating words (0.01 to 5.0, default: 1.2)
- `top_p`: Another way to control text creation (0.01 to 0.99, default: 0.8)

## Tips

- Include both an image and a text question for best results.
- Try different settings to change how the AI responds.
- For more detailed or creative answers, adjust the `temperature` or `top_p` settings.

## More Information

This is an implementation of the model [HuggingFaceM4/Idefics3-8B-Llama3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

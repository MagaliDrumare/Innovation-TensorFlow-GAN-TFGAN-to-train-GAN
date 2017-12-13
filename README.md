
# What is Tensorflow GAN ? 

### GAN For Beginners 
* https://www.oreilly.com/learning/generative-adversarial-networks-for-beginners

### TFGAN is a new library for training and evaluating Generative Adversarial Networks (GANs)- december 2017
* TFGAN, a lightweight Library for Generative Adversarial Networks: http://bit.ly/2z6yda3
* On GitHub:  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/README.md

### TFGAN Tutorial 
* Need to update Tensorflow to the last version 1.4.1 : pip install --upgrade tensorflow 
* https://github.com/tensorflow/models/blob/master/research/gan/tutorial.ipynb

# Step 1 : Create the Generator and the Discriminator 
```python 
// Create the generator 
def generator_fn(noise, weight_decay=2.5e-5):
    """Simple generator to produce MNIST images.
    
    Args:
        noise: A single Tensor representing noise.
        weight_decay: The value of the l2 weight decay.
    
    Returns:
        A generated image in the range [-1, 1].
    """
    with slim.arg_scope(
        [layers.fully_connected, layers.conv2d_transpose],
        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
        weights_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.fully_connected(noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 256)
        net = tf.reshape(net, [-1, 7, 7, 256])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net

// Create the discriminator 
def discriminator_fn(img, unused_conditioning, weight_decay=2.5e-5):
    """Discriminator network on MNIST digits.
    
    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
    
    Returns:
        Logits for the probability that the image is real.
    """
    with slim.arg_scope(
        [layers.conv2d, layers.fully_connected],
        activation_fn=leaky_relu, normalizer_fn=None,
        weights_regularizer=layers.l2_regularizer(weight_decay),
        biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.batch_norm)
        return layers.linear(net, 1)
```
  
# Step 2 : Create the GAN MODEL with tfgan.gan_model      
```python 
noise_dims = 64
gan_model = tfgan.gan_model(
    generator_fn,
    discriminator_fn,
    real_data=images,
    generator_inputs=tf.random_normal([batch_size, noise_dims]))

// Visualize the data generated with tfgan.eval.image_reshaper
generated_data_to_visualize = tfgan.eval.image_reshaper(
gan_model.generated_data[:20,...], num_cols=10)
visualize_digits(generated_data_to_visualize)  
```
# Step 3 : Loss Function with tfgan.gan_loss
```python 
# We can use the minimax loss from the original paper.
vanilla_gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=tfgan.losses.minimax_generator_loss,
    discriminator_loss_fn=tfgan.losses.minimax_discriminator_loss)

# We can use the Wasserstein loss (https://arxiv.org/abs/1701.07875) with the 
# gradient penalty from the improved Wasserstein loss paper 
# (https://arxiv.org/abs/1704.00028).
improved_wgan_loss = tfgan.gan_loss(
    gan_model,
    # We make the loss explicit for demonstration, even though the default is 
    # Wasserstein loss.
    generator_loss_fn=tfgan.losses.wasserstein_generator_loss,
    discriminator_loss_fn=tfgan.losses.wasserstein_discriminator_loss,
    gradient_penalty_weight=1.0)

# We can also define custom losses to use with the rest of the TFGAN framework.
def silly_custom_generator_loss(gan_model, add_summaries=False):
    return tf.reduce_mean(gan_model.discriminator_gen_outputs)
def silly_custom_discriminator_loss(gan_model, add_summaries=False):
    return (tf.reduce_mean(gan_model.discriminator_gen_outputs) -
            tf.reduce_mean(gan_model.discriminator_real_outputs))
custom_gan_loss = tfgan.gan_loss(
    gan_model,
    generator_loss_fn=silly_custom_generator_loss,
    discriminator_loss_fn=silly_custom_discriminator_loss)

# Sanity check that we can evaluate our losses.
for gan_loss, name in [(vanilla_gan_loss, 'vanilla loss'), 
                       (improved_wgan_loss, 'improved wgan loss'), 
                       (custom_gan_loss, 'custom loss')]:
    evaluate_tfgan_loss(gan_loss, name)
```
# Step 4 : Optimizer tfgan.gan_train_ops
```python
generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    gan_model,
    improved_wgan_loss,
    generator_optimizer,
    discriminator_optimizer)
    ```
# Step 5 : Evaluation of the model with the method generator_fn
```python 
num_images_to_eval = 500
MNIST_CLASSIFIER_FROZEN_GRAPH = './mnist/data/classify_mnist_graph_def.pb'

# For variables to load, use the same variable scope as in the train job.
with tf.variable_scope('Generator', reuse=True):
    eval_images = gan_model.generator_fn(
        tf.random_normal([num_images_to_eval, noise_dims]))
eval_score = util.mnist_score(eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)
```
# Final Step : Training with tfgan.get_sequential_train_steps
```python 
# We have the option to train the discriminator more than one step for every 
# step of the generator. In order to do this, we use a `GANTrainSteps` with 
# desired values. For this example, we use the default 1 generator train step 
# for every discriminator train step.
train_step_fn = tfgan.get_sequential_train_steps()

global_step = tf.train.get_or_create_global_step()
loss_values, mnist_score_values  = [], []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    with slim.queues.QueueRunners(sess):
        start_time = time.time()
        for i in xrange(801):
            cur_loss, _ = train_step_fn(
                sess, gan_train_ops, global_step, train_step_kwargs={})
            loss_values.append((i, cur_loss))
            if i % 100 == 0:
                mnist_score_values.append((i, sess.run(eval_score)))
            if i % 200 == 0:
                print('Current loss: %f' % cur_loss)
                print('Current MNIST score: %f' % mnist_score_values[-1][1])
                visualize_training_generator(
                    i, start_time, sess.run(generated_data_to_visualize))
```

# Results : Generated mnist images 
![alt tag](https://github.com/MagaliDrumare/Innovation-TensorFlow-GAN-TFGAN-to-train-GAN/blob/master/Generated%20mnist%20images%20.png)





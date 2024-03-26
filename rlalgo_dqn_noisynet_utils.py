def _reset_noisy(
    critic_head,
    tar_critic_head
):
    # NoisyNet -> reset noise 
    critic_head.reset_noise()
    tar_critic_head.reset_noise()
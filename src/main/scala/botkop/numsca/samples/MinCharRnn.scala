package botkop.numsca.samples

import botkop.numsca.Tensor
import botkop.{numsca => ns}

import scala.collection.mutable.ListBuffer
import scala.io.Source

/**
  * Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
  * Numsca implementation of https://gist.github.com/karpathy/d4dee566867f8291f086
  */
object MinCharRnn {

  def main(args: Array[String]): Unit = {
    // data I/O
    val data = Source.fromFile(args.head).toList
    val chars = data.toSet.toList
    val data_size = data.length
    val vocab_size = chars.length
    println(f"data has $data_size%d characters, $vocab_size%d unique.")
    val char_to_ix = chars.zipWithIndex.toMap
    val ix_to_char = char_to_ix.map(_.swap)

    // hyperparameters
    val hidden_size = 100 // size of hidden layer of neurons
    val seq_length = 25 // number of steps to unroll the RNN for
    val learning_rate = 1e-1

    // model parameters
    val Wxh = ns.randn(hidden_size, vocab_size) * 0.01 // input to hidden
    val Whh = ns.randn(hidden_size, hidden_size) * 0.01 // hidden to hidden
    val Why = ns.randn(vocab_size, hidden_size) * 0.01 // hidden to output
    val bh = ns.zeros(hidden_size, 1) // hidden bias
    val by = ns.zeros(vocab_size, 1) // output bias

    /**
      * inputs,targets are both list of integers.
      * hprev is Hx1 array of initial hidden state
      * returns the loss, gradients on model parameters, and last hidden state
      */
    def lossFun(inputs: List[Int], targets: List[Int], hprev: Tensor) = {
      val (xs, hs, ys, ps) = (new Array[Tensor](inputs.length),
                              new Array[Tensor](inputs.length),
                              new Array[Tensor](inputs.length),
                              new Array[Tensor](inputs.length))
      hs(hs.length - 1) = ns.copy(hprev)
      var loss = 0.0
      // forward pass
      for (t <- inputs.indices) {
        xs(t) = ns.zeros(vocab_size, 1) // encode in 1-of-k representation
        xs(t)(inputs(t), 0) := 1
        val it = if (t > 0) t else inputs.length // mimic python logic for negative indices
        hs(t) = ns.tanh(ns.dot(Wxh, xs(t)) + ns.dot(Whh, hs(it - 1) + bh)) // hidden state
        ys(t) = ns.dot(Why, hs(t)) + by // unnormalized log probabilities for next chars
        ps(t) = ns.exp(ys(t)) / ns.sum(ns.exp(ys(t))) // probabilities for next chars
        loss += -ns
          .log(ps(t)(targets(t), 0))
          .squeeze() // softmax (cross-entropy loss)
      }
      //backward pass: compute gradients going backwards
      val (dWxh, dWhh, dWhy) =
        (ns.zerosLike(Wxh), ns.zerosLike(Whh), ns.zerosLike(Why))
      val (dbh, dby) = (ns.zerosLike(bh), ns.zerosLike(by))
      val dhnext = ns.zerosLike(hs.head)
      for (t <- inputs.indices.reverse) {
        val dy = ns.copy(ps(t))
        dy(targets(t), 0) -= 1 // backprop into y. see http://cs231n.github.io/neural-networks-case-study///grad if confused here
        dWhy += ns.dot(dy, hs(t).T)
        dby += dy
        val dh = ns.dot(Why.T, dy) + dhnext // backprop into h
        val dhraw = (1 - hs(t) * hs(t)) * dh // backprop through tanh nonlinearity
        dbh += dhraw
        dWxh += ns.dot(dhraw, xs(t).T)

        val it = if (t > 0) t else inputs.length // mimic python logic negative indices
        dWhh += ns.dot(dhraw, hs(it - 1).T)
        dhnext := ns.dot(Whh.T, dhraw)
      }
      for (dparam <- List(dWxh, dWhh, dWhy, dbh, dby)) {
        dparam := ns.clip(dparam, -5, 5) // clip to mitigate exploding gradients
      }
      (loss, dWxh, dWhh, dWhy, dbh, dby, hs(inputs.length - 1))
    }

    /**
      * sample a sequence of integers from the model
      * h is memory state, seed_ix is seed letter for first time step
      */
    def sample(h: Tensor, seed_ix: Int, n: Int): ListBuffer[Int] = {
      val x = ns.zeros(vocab_size, 1)
      x(seed_ix, 0) := 1
      val ixes = ListBuffer.empty[Int]
      for (t <- 0 until n) {
        h := ns.tanh(ns.dot(Wxh, x) + ns.dot(Whh, h) + bh)
        val y = ns.dot(Why, h) + by
        val p = ns.exp(y) / ns.sum(ns.exp(y))
        val ix = ns.choice(ns.arange(vocab_size), p = p.ravel()).squeeze().toInt
        x := ns.zeros(vocab_size, 1)
        x(ix, 0) := 1
        ixes += ix
      }
      ixes
    }

    var (n, p) = (0, 0)
    val (mWxh, mWhh, mWhy) =
      (ns.zerosLike(Wxh), ns.zerosLike(Whh), ns.zerosLike(Why))
    val (mbh, mby) = (ns.zerosLike(bh), ns.zerosLike(by)) // memory variables for Adagrad
    var smooth_loss = -math.log(1.0 / vocab_size) * seq_length // loss at iteration 0

    val hprev = ns.zeros(hidden_size, 1)

    while (true) {
      // prepare inputs (we're sweeping from left to right in steps seq_length long)
      if (p + seq_length + 1 >= data.length || n == 0) {
        hprev := ns.zeros(hidden_size, 1) // reset RNN memory
        p = 0 // go from start of data
      }
      val inputs = data.slice(p, p + seq_length).map(char_to_ix)
      val targets = data.slice(p + 1, p + seq_length + 1).map(char_to_ix)

      // sample from the model now and then
      if (n % 100 == 0) {
        val sample_ix = sample(hprev, inputs.head, 200)
        val txt = sample_ix.map(t => ix_to_char(t)).mkString
        println(s"----\n$txt\n----")
      }

      // forward seq_length characters through the net and fetch gradient
      val (loss, dWxh, dWhh, dWhy, dbh, dby, hp) =
        lossFun(inputs, targets, hprev)
      hprev := hp
      smooth_loss = smooth_loss * 0.999 + loss * 0.001
      if (n % 100 == 0) println(s"iter $n, loss $smooth_loss") // print progress

      // perform parameter update with Adagrad
      (List(Wxh, Whh, Why, bh, by),
       List(dWxh, dWhh, dWhy, dbh, dby),
       List(mWxh, mWhh, mWhy, mbh, mby)).zipped.foreach {
        case (param, dparam, mem) =>
          mem += dparam * dparam
          param += -learning_rate * dparam / ns.sqrt(mem + 1e-8) // adagrad update
      }

      p += seq_length // move data pointer
      n += 1 // iteration counter
    }
  }

}

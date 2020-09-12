
            # inputs projected by all gate matrices at once
        matrix_x = K.dot(inputs, self.kernel)
        if self.use_bias:
                # biases: bias_z_i, bias_r_i, bias_h_i
            matrix_x = K.bias_add(matrix_x, self.input_bias)
        x_z = matrix_x[:, :self.units]
        x_r = matrix_x[:, self.units: 2 * self.units]
        x_h = matrix_x[:, 2 * self.units:]


                # hidden state projected by all gate matrices at once
        matrix_inner = K.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            matrix_inner = K.bias_add(matrix_inner, self.recurrent_bias)

        recurrent_z = matrix_inner[:, :self.units]
        recurrent_r = matrix_inner[:, self.units: 2 * self.units]

        z = self.recurrent_activation(x_z + recurrent_z)
        r = self.recurrent_activation(x_r + recurrent_r)

        recurrent_h = r * matrix_inner[:, 2 * self.units:]

        hh = self.activation(x_h + recurrent_h)

        # previous and candidate state mixed by update gate
        h = z * h_tm1 + (1 - z) * hh

        return h, [h]

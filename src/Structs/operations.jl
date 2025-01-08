# define equality and approximate equality for custom types
# isapprox is only used for numeric structs that will serve as a source of floating point error, the uses of "==" in its definition are intentional
for op in [==, isapprox]
    @eval begin
        function $op(e1::EncodedTimeSeriesSet, e2::EncodedTimeSeriesSet)
            return e1.class_distribution, e2.class_distribution && $op(e1.original_data, e2.original_data) && $op(e1.timeseries, e2.timeseries)
        end

        function $op(p1::PState, p2::PState) 
            return p1.label == p2.label && p1.label_index == p2.label_index && $op(p1.pstate.data, p2.pstate.data)
        end

        function $op(m1::TrainedMPS, m2::TrainedMPS)

            return m1.opts == m2.opts && $op(m1.train_data, m2.train_data) && $op(m1.mps.data, m2.mps.data)
        end
        
    end
end

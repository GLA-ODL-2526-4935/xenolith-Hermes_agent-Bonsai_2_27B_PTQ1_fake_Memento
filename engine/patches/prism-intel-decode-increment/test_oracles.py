"""Independent integer/memory-contract oracles; not native kernel parity."""
import random
import unittest

class Oracles(unittest.TestCase):
    def test_all_pair_tables(self):
        for a in range(-128,128):
            for b in range(-128,128):
                algebra=[-a-b,-b,a-b,-a,0,a,b-a,b,a+b]
                reference=[i*a+j*b for j in (-1,0,1) for i in (-1,0,1)]
                self.assertEqual(algebra,reference)
                self.assertTrue(all(-256<=v<=256 for v in algebra))
                # Low/high byte reconstruction must sign-extend, not saturate.
                for value in algebra:
                    bits=((value>>8)&255)*256+(value&255)
                    self.assertEqual(bits-65536 if bits&32768 else bits,value)

    def test_int16_group_bound_and_scale_partition(self):
        self.assertLessEqual(16*256,32767)
        self.assertGreaterEqual(-16*256,-32768)
        rng=random.Random(42)
        for _ in range(2000):
            weights=[rng.randrange(-1,2) for _ in range(128)]
            activations=[rng.randrange(-128,128) for _ in range(128)]
            reference=[sum(weights[i]*activations[i] for i in range(g*32,g*32+32)) for g in range(4)]
            paired=[sum(weights[2*p]*activations[2*p]+weights[2*p+1]*activations[2*p+1]
                        for p in range(g*16,g*16+16)) for g in range(4)]
            self.assertEqual(reference,paired)
            self.assertTrue(all(-4096<=s<=4096 for s in paired))

    def test_all_tail_encodings(self):
        for packed in range(65536):
            values=[packed&255,packed>>8]
            scalar=[]
            for j in range(8):
                state=values[j%2]
                for _ in range(j//2): state=(state*3)&255
                scalar.append((state*3)>>8)
            multiplied=[(((values[j%2]*(3**(j//2)))&255)*3)>>8 for j in range(8)]
            self.assertEqual(scalar,multiplied)

    def test_capture_acceptance_contract(self):
        for count in (1,2,3,5,128,513):
            source=[bytes([i%256])*28 for i in range(count)]
            earlier=source[:-1]
            pending=source[-1]
            for accepted in range(count+2):
                selected=min(accepted,count-1)
                actual=pending if selected==count-1 else earlier[selected]
                self.assertEqual(actual,source[selected])
            # bytes copied: count rows rather than count+1; no KV/state claim.
            self.assertEqual(len(earlier)+1,count)

if __name__=='__main__': unittest.main()
